from __future__ import annotations

import os
from pathlib import Path

import folder_paths

from ...constants import CATEGORY_IMAGE
from ..core.image import tensor2pil
from .helpers import Input
from .image_save import sanitize_filename_part

PREVIEW_DELIMITER = "_"
PREVIEW_NUMBER_PADDING = 4
PREVIEW_EXTENSION = "png"
DEFAULT_PREVIEW_IMAGE_NAME = "preview"


class ImagePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "images": ("IMAGE",),
            },
            optional={
                "info": ("PIPELINE_INFO",),
            },
        )

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = CATEGORY_IMAGE
    OUTPUT_NODE = True

    def preview(self, images, info=None):
        return preview_images(images, info.name if info else None)


def preview_images(images, name: str | None = None):
    output_dir = folder_paths.get_temp_directory()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename_prefix = _preview_filename_prefix(name)
    counter = _get_next_counter(output_dir, filename_prefix, PREVIEW_EXTENSION)
    results = []

    for image in images:
        filename = _next_available_filename(output_dir, filename_prefix, counter, PREVIEW_EXTENSION)
        output_file = os.path.join(output_dir, filename)

        pil_image = tensor2pil(image)
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        pil_image.save(output_file, compress_level=4)

        results.append({"filename": filename, "subfolder": "", "type": "temp"})
        counter += 1

    return {"ui": {"images": results}}


def _preview_filename_prefix(name: str | None) -> str:
    if name:
        prefix = sanitize_filename_part(name)
        if prefix:
            return prefix
    return DEFAULT_PREVIEW_IMAGE_NAME


def _next_available_filename(output_dir: str, filename_prefix: str, counter: int, extension: str) -> str:
    while True:
        filename = f"{filename_prefix}{PREVIEW_DELIMITER}{counter:0{PREVIEW_NUMBER_PADDING}}.{extension}"
        if not os.path.exists(os.path.join(output_dir, filename)):
            return filename
        counter += 1


def _get_next_counter(output_dir: str, filename_prefix: str, extension: str) -> int:
    counters = []
    prefix = f"{filename_prefix}{PREVIEW_DELIMITER}"
    suffix = f".{extension}"

    try:
        filenames = os.listdir(output_dir)
    except FileNotFoundError:
        return 1

    for filename in filenames:
        if not filename.startswith(prefix) or not filename.endswith(suffix):
            continue

        number = filename[len(prefix) : -len(suffix)]
        if number.isdigit():
            counters.append(int(number))

    return max(counters, default=0) + 1
