from __future__ import annotations

from pathlib import Path

import comfy.model_management
import comfy.model_patcher
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from ...constants import (
    BIREFNET_INFERENCE_MEMORY,
    BIREFNET_INPUT_SIZE,
    BIREFNET_MODEL_RELATIVE_PATH,
    CATEGORY_REMBG,
    DEFAULT_REMBG_BLUR,
    DEFAULT_REMBG_CONTRAST,
    DEFAULT_REMBG_SHARPEN_ITERATIONS,
    DEFAULT_REMBG_SHARPEN_KERNEL,
    REMBG_BLUR_MAX,
    REMBG_BLUR_MIN,
    REMBG_CONTRAST_MAX,
    REMBG_CONTRAST_MIN,
    REMBG_CONTRAST_STEP,
    REMBG_SHARPEN_MAX,
    REMBG_SHARPEN_MIN,
)
from ..core.image import pil2tensor, tensor2pil, to_mask_tensor
from .helpers import Input
from .image_filters import ImageAdjustment, ImageLucySharpen, image_input

BIREFNET_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
BIREFNET_NORMALIZE_STD = [0.229, 0.224, 0.225]


def birefnet_model_path() -> Path:
    extension_root = Path(__file__).resolve().parents[2]
    return extension_root.joinpath(*BIREFNET_MODEL_RELATIVE_PATH)


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}

    for key, value in state_dict.items():
        key = key.removeprefix("model.").removeprefix("module.")
        normalized[key] = value

    return normalized


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    min_value = torch.min(mask)
    max_value = torch.max(mask)

    if max_value == min_value:
        return torch.zeros_like(mask)

    return (mask - min_value) / (max_value - min_value)


def prediction_from_output(output):
    while isinstance(output, (list, tuple)):
        output = output[-1]

    return output


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
        size=size,
        mode="bilinear",
    ).squeeze(1)


def apply_alpha(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if len(images) != len(masks):
        raise ValueError(f"Image and mask batch sizes must match: {len(images)} != {len(masks)}")

    masks = resize_mask(masks, images.shape[1:3])
    rgba_images = [torch.cat((images[index][:, :, :3], masks[index].unsqueeze(2)), dim=2) for index in range(len(images))]

    return torch.stack(rgba_images)


class BiRefNetModel:
    """The weights behind a ModelPatcher, so ComfyUI loads them when asked and can evict them when it needs room.

    They used to be moved to "cuda" once and left there: ~885 MB the model manager could neither see nor free,
    and a second copy for every node instance, since BiRefNetRemBGAdvanced builds its own BiRefNetRemBG.
    """
    def __init__(self):
        self.patcher = None
        self.model_path = birefnet_model_path()
        self.transform = transforms.Compose(
            [
                transforms.Resize(BIREFNET_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(BIREFNET_NORMALIZE_MEAN, BIREFNET_NORMALIZE_STD),
            ]
        )

    def remove_background(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.ensure_loaded()
        comfy.model_management.load_models_gpu([self.patcher], memory_required=BIREFNET_INFERENCE_MEMORY, force_full_load=True)

        output_images = []
        output_masks = []

        for image in images:
            original = tensor2pil(image).convert("RGB")
            mask = self.predict_mask(original)

            rgba = original.convert("RGBA")
            rgba.putalpha(mask)

            output_images.append(pil2tensor(rgba))
            output_masks.append(to_mask_tensor(mask))

        return torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)

    def ensure_loaded(self):
        if self.patcher is not None:
            return

        if not self.model_path.is_file():
            raise FileNotFoundError(f"BiRefNet weights not found: {self.model_path}")

        try:
            from ..core.rembg import BiRefNet, BiRefNetConfig
        except ImportError as error:
            raise ImportError(
                "Missing BiRefNet dependency. Install requirements.txt for comfyui-ne-tools. "
                "Required packages include: timm, kornia, transformers, safetensors."
            ) from error

        model = BiRefNet(config=BiRefNetConfig(bb_pretrained=False))
        state_dict = normalize_state_dict(load_file(str(self.model_path), device="cpu"))

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"BiRefNet unexpected keys: {len(unexpected)}")
        if missing:
            print(f"BiRefNet missing keys: {len(missing)}")

        model.eval()
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            _Holder(model),
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )

    def predict_mask(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        input_tensor = self.transform(image).unsqueeze(0).to(self.patcher.load_device)

        with torch.no_grad():
            output = self.patcher.model(input_tensor)

        prediction = F.interpolate(
            prediction_from_output(output),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        mask = normalize_mask(prediction.sigmoid().squeeze().detach().cpu())
        return transforms.ToPILImage()(mask)


class _Holder(torch.nn.Module):
    """A plain module around the network: the patcher assigns `device` on what it holds, and BiRefNet's is a read-only property."""
    def __init__(self, network: torch.nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, image: torch.Tensor):
        return self.network(image)


# One set of weights for every node that removes a background.
_birefnet = BiRefNetModel()


class BiRefNetRemBG:
    def __init__(self):
        self.model = _birefnet

    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(images=image_input())

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "execute"
    CATEGORY = CATEGORY_REMBG

    def execute(self, images):
        return self.model.remove_background(images)


class BiRefNetRemBGAdvanced:
    def __init__(self):
        self.rembg = BiRefNetRemBG()
        self.sharpen = ImageLucySharpen()
        self.adjust = ImageAdjustment()

    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            images=image_input(),
            sharpen_iterations=Input.int(DEFAULT_REMBG_SHARPEN_ITERATIONS, REMBG_SHARPEN_MIN, REMBG_SHARPEN_MAX),
            sharpen_kernel=Input.int(DEFAULT_REMBG_SHARPEN_KERNEL, REMBG_SHARPEN_MIN, REMBG_SHARPEN_MAX),
            contrast=Input.float(DEFAULT_REMBG_CONTRAST, REMBG_CONTRAST_MIN, REMBG_CONTRAST_MAX, REMBG_CONTRAST_STEP),
            blur=Input.int(DEFAULT_REMBG_BLUR, REMBG_BLUR_MIN, REMBG_BLUR_MAX),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_REMBG

    def execute(self, images, sharpen_iterations, sharpen_kernel, contrast, blur):
        processed_images = self.sharpen.execute(images, sharpen_iterations, sharpen_kernel)[0]
        processed_images = self.adjust.execute(
            processed_images,
            brightness=0.0,
            contrast=round(contrast, 2),
            saturation=1.0,
            sharpness=1.0,
            blur=blur,
            gaussian_blur=0.0,
            edge_enhance=0.0,
            detail_enhance=True,
        )[0]

        _, masks = self.rembg.execute(processed_images)
        return (apply_alpha(images, masks),)
