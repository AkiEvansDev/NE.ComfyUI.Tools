from __future__ import annotations

import numpy as np
import torch
from nodes import ImageScale, LoadImage, SetLatentNoiseMask, VAEDecode
from PIL import Image

from ...constants import *
from ..core.image import ensure_mask_batch, pil2tensor, scale_mask
from .helpers import Input, encode_image_to_latent, parse_dimensions, round_float


def normalize_dimension(value: int) -> int:
    return max(IMAGE_SIZE_MIN, (value // 8) * 8)


class ImageBlank:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            width=Input.int(DEFAULT_IMAGE_WIDTH, IMAGE_SIZE_MIN, IMAGE_SIZE_MAX),
            height=Input.int(DEFAULT_IMAGE_HEIGHT, IMAGE_SIZE_MIN, IMAGE_SIZE_MAX),
            red=Input.int(DEFAULT_IMAGE_CHANNEL, IMAGE_CHANNEL_MIN, IMAGE_CHANNEL_MAX),
            green=Input.int(DEFAULT_IMAGE_CHANNEL, IMAGE_CHANNEL_MIN, IMAGE_CHANNEL_MAX),
            blue=Input.int(DEFAULT_IMAGE_CHANNEL, IMAGE_CHANNEL_MIN, IMAGE_CHANNEL_MAX),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blank_image"
    CATEGORY = CATEGORY_IMAGE

    def blank_image(self, width: int, height: int, red: int, green: int, blue: int):
        image = Image.new(
            mode="RGB",
            size=(normalize_dimension(width), normalize_dimension(height)),
            color=(red, green, blue),
        )
        return (pil2tensor(image),)


class ImagePowerNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            width=Input.int(DEFAULT_IMAGE_WIDTH, IMAGE_SIZE_MIN, IMAGE_SIZE_MAX),
            height=Input.int(DEFAULT_IMAGE_HEIGHT, IMAGE_SIZE_MIN, IMAGE_SIZE_MAX),
            attenuation=Input.float(
                DEFAULT_POWER_NOISE_ATTENUATION,
                POWER_NOISE_ATTENUATION_MIN,
                POWER_NOISE_ATTENUATION_MAX,
                POWER_NOISE_ATTENUATION_STEP,
            ),
            noise_type=Input.choice(POWER_NOISE_TYPES, DEFAULT_POWER_NOISE_TYPE),
            seed=Input.int(0, 0, POWER_NOISE_MAX_SEED),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "power_noise"
    CATEGORY = CATEGORY_IMAGE

    def power_noise(self, width: int, height: int, attenuation: float, noise_type: str, seed: int):
        image = self.generate_power_noise(
            normalize_dimension(width),
            normalize_dimension(height),
            round_float(attenuation),
            noise_type,
            seed,
        )
        return (pil2tensor(image),)

    @staticmethod
    def _normalize_noise(noise: np.ndarray) -> np.ndarray:
        min_value = np.min(noise)
        max_value = np.max(noise)

        if max_value == min_value:
            return np.zeros_like(noise)

        return 255 * (noise - min_value) / (max_value - min_value)

    def generate_power_noise(self, width: int, height: int, attenuation: float, noise_type: str, seed: int) -> Image.Image:
        rng = np.random.default_rng(seed)

        def grey_noise() -> np.ndarray:
            return rng.normal(0, attenuation, (height, width))

        def white_noise() -> np.ndarray:
            return rng.random((height, width))

        def spectral_noise(power_modifier) -> np.ndarray:
            noise = grey_noise()
            scale = 1.0 / (width * height)
            fy = np.fft.fftfreq(height)[:, np.newaxis] ** 2
            fx = np.fft.fftfreq(width) ** 2
            power = np.sqrt(fy + fx)
            power[0, 0] = 1
            noise = np.fft.ifft2(power_modifier(np.fft.fft2(noise), power))
            noise *= scale / noise.std()
            return np.real(noise)

        # `power` is the radial frequency, so these shape the amplitude and the power spectrum is their square:
        # pink falls as 1/f, blue rises as f, green is a band in the middle. The formulas carried over from the
        # WAS suite had it the other way round - "pink" was violet, "blue" was brown and "green" was pink.
        generators = {
            "white": white_noise,
            "grey": grey_noise,
            "blue": lambda: spectral_noise(lambda fft, power: fft * np.sqrt(power)),
            "green": lambda: spectral_noise(
                lambda fft, power: fft * np.exp(-(((power - POWER_NOISE_GREEN_CENTER) / POWER_NOISE_GREEN_WIDTH) ** 2)),
            ),
            "pink": lambda: spectral_noise(lambda fft, power: fft / np.sqrt(power)),
        }

        def typed_noise(selected_type: str) -> np.ndarray:
            return generators.get(selected_type, white_noise)()

        if noise_type == "mix":
            noise = self._blend_noise(width, height, seed, typed_noise)
        else:
            noise = self._normalize_noise(typed_noise(noise_type))

        return Image.fromarray(noise.astype(np.uint8).real).convert("RGB")

    def _blend_noise(self, width: int, height: int, seed: int, typed_noise) -> np.ndarray:
        blended_image = Image.new("L", (width, height), color=0)
        noise_types = ["white", "grey", "pink", "green", "blue"]

        for index, selected_type in enumerate(noise_types):
            mask_rng = np.random.default_rng(seed + index)
            mask = mask_rng.normal(0, 0.5, (height, width))
            mask_image = Image.fromarray(self._normalize_noise(mask).astype(np.uint8).real)
            noise_image = Image.fromarray(self._normalize_noise(typed_noise(selected_type)).astype(np.uint8).real)
            blended_image = Image.composite(blended_image, noise_image, mask_image)

        return np.asarray(blended_image)


class ImageToLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "dimensions": Input.choice(DIMENSIONS, DEFAULT_DIMENSIONS),
                "mode": Input.choice(IMAGE_TO_LATENT_MODES, DEFAULT_IMAGE_TO_LATENT_MODE),
                "resize_method": Input.choice(IMAGE_RESIZE_METHODS, DEFAULT_IMAGE_RESIZE_METHOD),
            },
            optional={
                "mask": ("MASK",),
                "invert_mask": Input.bool(False),
            },
        )

    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("latent", "image", "mask", "width", "height")
    FUNCTION = "encode"
    CATEGORY = CATEGORY_IMAGE

    def encode(self, image, vae, dimensions: str, mode: str, resize_method: str, mask=None, invert_mask: bool = False):
        target = self._target_size(image, dimensions, mode)
        crop = "center" if mode == IMAGE_TO_LATENT_MODE_COVER else "disabled"

        image = self._fit_image(image, target, crop, resize_method)
        latent = encode_image_to_latent(vae, image)
        mask = self._fit_mask(mask, target, crop, resize_method, invert_mask)

        if mask is not None:
            latent, = SetLatentNoiseMask().set_mask(latent, mask)
        else:
            # An all-ones mask means "everything may change", which is what the unmasked run does.
            mask = torch.ones((1, target[1], target[0]), dtype=torch.float32)

        return (latent, image, mask, target[0], target[1])

    @staticmethod
    def _target_size(image, dimensions: str, mode: str) -> tuple[int, int]:
        if mode == IMAGE_TO_LATENT_MODE_SOURCE:
            _, height, width, _ = image.shape
            return normalize_dimension(width), normalize_dimension(height)

        return parse_dimensions(dimensions)

    @staticmethod
    def _fit_image(image, target: tuple[int, int], crop: str, resize_method: str):
        _, height, width, _ = image.shape
        if (width, height) == target:
            return image

        result, = ImageScale().upscale(image, resize_method, target[0], target[1], crop)
        return result

    @staticmethod
    def _fit_mask(mask, target: tuple[int, int], crop: str, resize_method: str, invert_mask: bool):
        if mask is None:
            return None

        mask = ensure_mask_batch(mask)
        _, height, width = mask.shape

        if (width, height) != target:
            # The image's own crop, otherwise a cropped source and its mask stop pointing at the same pixels.
            mask = scale_mask(mask, target[0], target[1], resize_method, crop)

        if invert_mask:
            mask = 1.0 - mask

        return mask


class LatentToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            latent=("LATENT",),
            vae=("VAE",),
        )

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "decode"
    CATEGORY = CATEGORY_IMAGE

    def decode(self, latent, vae):
        image, = VAEDecode().decode(vae, latent)
        _, height, width, _ = image.shape
        return (image, width, height)


class LoadImageFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(image=Input.force("STRING"))

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    CATEGORY = CATEGORY_IMAGE

    def load_image(self, image: str):
        return LoadImage().load_image(image)

    @classmethod
    def IS_CHANGED(cls, image: str | None):
        if image is None:
            return 0

        return LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image: str | None):
        if image is None:
            return True

        return LoadImage.VALIDATE_INPUTS(image)
