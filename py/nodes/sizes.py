from __future__ import annotations

from ...constants import CATEGORY_SIZE
from .helpers import Input


class GetLatentSize:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            latent=("LATENT",),
            original=Input.bool(False),
        )

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = CATEGORY_SIZE

    def get_size(self, latent, original: bool):
        samples = latent["samples"]
        width = samples.shape[3]
        height = samples.shape[2]

        if not original:
            width *= 8
            height *= 8

        return (width, height)


class GetImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(image=("IMAGE",))

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = CATEGORY_SIZE

    def get_size(self, image):
        return (image.shape[2], image.shape[1])
