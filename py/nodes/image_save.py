from __future__ import annotations

import json
import os
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import comfy.model_management
import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ...constants import *
from ..core.image import tensor2pil
from .helpers import Input, format_info, pipeline_info


@dataclass(frozen=True, slots=True)
class SaveImageOptions:
    folder: str
    prefix: str
    delimiter: str
    number_padding: int
    number_start: bool
    extension: str
    dpi: int
    quality: int
    optimize_image: bool
    lossless_webp: bool
    embed_workflow: bool
    save_info: bool


class TextTokens:
    def __init__(self) -> None:
        device = comfy.model_management.get_torch_device()
        self.tokens = {
            "[hostname]": socket.gethostname(),
            "[cuda_device]": str(device),
            "[cuda_name]": str(comfy.model_management.get_torch_device_name(device=device)),
        }

        try:
            self.tokens["[user]"] = os.getlogin() or "null"
        except Exception:
            self.tokens["[user]"] = "null"

    def parse(self, text: str | None) -> str | None:
        if text is None:
            return None

        tokens = self.tokens | {
            "[time]": str(time.time()).split(".", 1)[0],
            "[date]": time.strftime(DEFAULT_SAVE_DATE_FORMAT),
        }

        for token, value in tokens.items():
            text = re.sub(re.escape(token), value, text)

        return re.sub(r"\[time\((.*?)\)\]", lambda match: time.strftime(match.group(1)), text)


def sanitize_filename_part(value: Any, max_length: int = 100) -> str:
    safe = str(value).strip()
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", safe)
    safe = safe.replace("..", "_")
    return safe[:max_length]


def normalize_output_extension(extension: str) -> str:
    extension = extension.lower().lstrip(".")
    file_extension = f".{extension}"

    if file_extension in ALLOWED_IMAGE_EXTENSIONS:
        return extension

    print(
        f"The extension `{extension}` is not valid. "
        f"The valid formats are: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
    )
    return DEFAULT_SAVE_EXTENSION


def build_output_subfolder(folder: str, tokens: TextTokens) -> str:
    """The folder widget as a path under the output directory — empty means the output root itself.

    Every segment goes through sanitize_filename_part, so no token expansion and no hand-written
    `..` or drive letter can climb out of the output directory.
    """
    segments = re.split(r"[\\/]+", tokens.parse(folder) or "")
    return "/".join(sanitize_filename_part(segment) for segment in segments if segment.strip(" ."))


def build_filename_prefix(prefix: str, name: str | None, delimiter: str, tokens: TextTokens) -> str:
    """The name pinned upstream (the seed) leads, the prefix widget follows; either half may be empty."""
    parts = [sanitize_filename_part(name) if name else "", sanitize_filename_part(tokens.parse(prefix) or "")]
    return delimiter.join(part for part in parts if part)


def next_counter(output_path: str, filename_prefix: str, delimiter: str, number_start: bool) -> int:
    if not filename_prefix:
        pattern = r"(\d+)"
    elif number_start:
        pattern = rf"(\d+){re.escape(delimiter)}{re.escape(filename_prefix)}"
    else:
        pattern = rf"{re.escape(filename_prefix)}{re.escape(delimiter)}(\d+)"

    counters = []
    for filename in os.listdir(output_path):
        match = re.match(pattern, os.path.basename(filename))
        if match:
            counters.append(int(match.group(1)))

    return max(counters, default=0) + 1


def next_available_filename(
    output_path: str,
    prefix: str,
    counter: int,
    *,
    delimiter: str,
    padding: int,
    extension: str,
    number_start: bool,
) -> str:
    file_extension = f".{extension}"

    # A prefix that already carries the extension is a literal file name: taken as is while it is free,
    # then it falls back to numbering rather than overwriting.
    if prefix.endswith(file_extension):
        if not os.path.exists(os.path.join(output_path, prefix)):
            return prefix

        prefix = prefix[: -len(file_extension)]

    while True:
        filename = build_numbered_filename(
            prefix=prefix,
            delimiter=delimiter,
            padding=padding,
            counter=counter,
            extension=extension,
            number_start=number_start,
        )

        if not os.path.exists(os.path.join(output_path, filename)):
            return filename

        counter += 1


def build_numbered_filename(
    *,
    prefix: str,
    delimiter: str,
    padding: int,
    counter: int,
    extension: str,
    number_start: bool,
) -> str:
    file_extension = f".{extension}"

    if prefix.endswith(file_extension):
        return prefix

    number = f"{counter:0{padding}}"
    if not prefix:
        return f"{number}{file_extension}"

    if number_start:
        return f"{number}{delimiter}{prefix}{file_extension}"

    return f"{prefix}{delimiter}{number}{file_extension}"


class CustomImageSave:
    def __init__(self) -> None:
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "active": Input.bool(True),
                "images": ("IMAGE",),
                "folder": Input.string(DEFAULT_SAVE_IMAGE_FOLDER),
                "prefix": Input.string(DEFAULT_SAVE_IMAGE_PREFIX),
                "delimiter": Input.string(DEFAULT_SAVE_DELIMITER),
                "number_padding": Input.int(
                    DEFAULT_SAVE_NUMBER_PADDING,
                    SAVE_NUMBER_PADDING_MIN,
                    SAVE_NUMBER_PADDING_MAX,
                ),
                "number_start": Input.bool(True),
                "extension": Input.choice(SAVE_IMAGE_EXTENSIONS, DEFAULT_SAVE_EXTENSION),
                "dpi": Input.int(DEFAULT_SAVE_DPI, SAVE_DPI_MIN, SAVE_DPI_MAX),
                "quality": Input.int(DEFAULT_SAVE_QUALITY, SAVE_QUALITY_MIN, SAVE_QUALITY_MAX),
                "optimize_image": Input.bool(False),
                "lossless_webp": Input.bool(False),
                "embed_workflow": Input.bool(False),
                "save_info": Input.bool(True),
            },
            optional={
                "info": ("PIPELINE_INFO",),
            },
            hidden={
                "workflow": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        )

    RETURN_TYPES = ("IMAGE", "PIPELINE_INFO")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "save_image"
    CATEGORY = CATEGORY_IMAGE

    def save_image(
        self,
        active: bool,
        images,
        folder: str,
        prefix: str,
        delimiter: str,
        number_padding: int,
        number_start: bool,
        extension: str,
        dpi: int,
        quality: int,
        optimize_image: bool,
        lossless_webp: bool,
        embed_workflow: bool,
        save_info: bool,
        info=None,
        workflow: Any = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ):
        options = SaveImageOptions(
            folder=folder,
            prefix=prefix,
            delimiter=delimiter,
            number_padding=number_padding,
            number_start=number_start,
            extension=normalize_output_extension(extension),
            dpi=dpi,
            quality=quality,
            optimize_image=optimize_image,
            lossless_webp=lossless_webp,
            embed_workflow=embed_workflow,
            save_info=save_info,
        )

        info = pipeline_info(info, self._build_block(active, options))

        if active:
            self._save_images(images, options, info.text, info.name, workflow, extra_pnginfo)

        return (images, info)

    @staticmethod
    def _build_block(active: bool, options: SaveImageOptions) -> str:
        return format_info(
            "Save Image",
            {
                "Active": active,
                "Extension": options.extension,
                "Dpi": options.dpi,
                "Quality": options.quality,
                "Optimize": options.optimize_image,
            },
        )

    def _save_images(
        self,
        images,
        options: SaveImageOptions,
        info: str,
        name: str | None,
        workflow: Any,
        extra_pnginfo: dict[str, Any] | None,
    ) -> None:
        tokens = TextTokens()
        subfolder = build_output_subfolder(options.folder, tokens)
        filename_prefix = build_filename_prefix(options.prefix, name, options.delimiter, tokens)

        output_path = os.path.join(self.output_dir, subfolder) if subfolder else self.output_dir
        Path(output_path).mkdir(parents=True, exist_ok=True)

        counter = next_counter(output_path, filename_prefix, options.delimiter, options.number_start)

        for image in images:
            filename = next_available_filename(
                output_path,
                filename_prefix,
                counter,
                delimiter=options.delimiter,
                padding=options.number_padding,
                extension=options.extension,
                number_start=options.number_start,
            )
            output_file = os.path.abspath(os.path.join(output_path, filename))

            # A failed save fails the run. Caught and printed, a full disk or a locked folder looked exactly like
            # a successful one and the image was gone.
            self._save_pil_image(
                tensor2pil(image),
                output_file,
                options,
                self._build_metadata(options.extension, options.embed_workflow, workflow, extra_pnginfo),
            )

            if options.save_info:
                Path(f"{output_file}.txt").write_text(info, encoding="utf-8")

            print(f"Image file saved to: {output_file}")
            counter += 1

    @staticmethod
    def _build_metadata(extension: str, embed_workflow: bool, workflow: Any, extra_pnginfo: dict[str, Any] | None):
        if extension == "webp":
            img_exif = Image.Exif()

            if embed_workflow:
                workflow_metadata = ""

                if workflow is not None:
                    img_exif[0x010F] = "Prompt:" + json.dumps(workflow)

                if extra_pnginfo is not None:
                    for key in extra_pnginfo:
                        workflow_metadata += json.dumps(extra_pnginfo[key])

                img_exif[0x010E] = "Workflow:" + workflow_metadata

            return img_exif.tobytes()

        metadata = PngInfo()

        if embed_workflow:
            if workflow is not None:
                metadata.add_text("prompt", json.dumps(workflow))

            if extra_pnginfo is not None:
                for key in extra_pnginfo:
                    metadata.add_text(key, json.dumps(extra_pnginfo[key]))

        return metadata

    @staticmethod
    def _save_pil_image(img: Image.Image, output_file: str, options: SaveImageOptions, metadata) -> None:
        extension = options.extension

        if extension in ("jpg", "jpeg", "bmp") and img.mode == "RGBA":
            img = img.convert("RGB")

        if extension in ("jpg", "jpeg"):
            img.save(output_file, quality=options.quality, optimize=options.optimize_image, dpi=(options.dpi, options.dpi))
        elif extension == "webp":
            img.save(output_file, quality=options.quality, lossless=options.lossless_webp, exif=metadata)
        elif extension == "png":
            img.save(output_file, pnginfo=metadata, optimize=options.optimize_image, dpi=(options.dpi, options.dpi))
        elif extension == "bmp":
            img.save(output_file)
        elif extension == "tiff":
            img.save(output_file, quality=options.quality, optimize=options.optimize_image)
        else:
            img.save(output_file, pnginfo=metadata, optimize=options.optimize_image)
