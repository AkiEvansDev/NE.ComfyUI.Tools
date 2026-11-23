from __future__ import annotations

from collections.abc import Callable

import comfy.utils
import numpy as np
import torch
from PIL import Image


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image: torch.Tensor) -> Image.Image:
    array = 255.0 * image.detach().cpu().numpy()
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def to_pil_rgb(image: torch.Tensor) -> Image.Image:
    return tensor2pil(image).convert("RGB")


def to_pil_rgba(image: torch.Tensor) -> Image.Image:
    return tensor2pil(image).convert("RGBA")


def to_tensor_image(image: Image.Image) -> torch.Tensor:
    return pil2tensor(image)


def to_mask_tensor(mask: Image.Image) -> torch.Tensor:
    return pil2tensor(mask).squeeze(-1)


def process_batch(images: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    return torch.cat([fn(image) for image in images], dim=0)


def ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def frame_at(tensor: torch.Tensor, index: int) -> torch.Tensor:
    """One frame of a batch, reusing the last when the batches are of different length.

    A single mask paired with a batch of images is the usual case, and it should cover all of them.
    """
    return tensor[min(index, tensor.shape[0] - 1)]


def scale_mask(mask: torch.Tensor, width: int, height: int, method: str, crop: str = "disabled") -> torch.Tensor:
    """Resize a (B, H, W) mask, matching how an image of the same geometry would be resized.

    Expanded to three channels first: comfy's lanczos path runs through PIL and mangles a
    single-channel tensor, returning the right mean with the wrong geometry.
    """
    mask = ensure_mask_batch(mask)
    expanded = mask.unsqueeze(1).repeat(1, 3, 1, 1)
    scaled = comfy.utils.common_upscale(expanded, width, height, method, crop)
    return scaled[:, 0].clamp(0.0, 1.0)


def ensure_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        return mask.squeeze(-1)
    return mask
