from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import folder_paths

from ...constants import *
from .helpers import Input, format_info, pipeline_info
from .image_save import TextTokens, build_filename_prefix, build_output_subfolder, next_available_filename, next_counter


@dataclass(frozen=True, slots=True)
class SaveVideoOptions:
    folder: str
    prefix: str
    delimiter: str
    number_padding: int
    number_start: bool
    extension: str
    quality: int
    save_info: bool


@dataclass(frozen=True, slots=True)
class SavedVideoFile:
    file: str
    subfolder: str
    full_path: str


def normalize_video_extension(extension: str) -> str:
    extension = extension.lower().lstrip(".")
    if extension in SAVE_VIDEO_EXTENSIONS:
        return extension

    print(
        f"The extension `{extension}` is not valid. "
        f"The valid formats are: {', '.join(SAVE_VIDEO_EXTENSIONS)}"
    )
    return DEFAULT_SAVE_VIDEO_EXTENSION


class CustomVideoSave:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "active": Input.bool(True),
                "images": ("IMAGE",),
                "video_config": ("VIDEO_CONFIG",),
                "folder": Input.string(DEFAULT_SAVE_VIDEO_FOLDER),
                "prefix": Input.string(DEFAULT_SAVE_VIDEO_PREFIX),
                "delimiter": Input.string(DEFAULT_SAVE_DELIMITER),
                "number_padding": Input.int(
                    DEFAULT_SAVE_NUMBER_PADDING,
                    SAVE_NUMBER_PADDING_MIN,
                    SAVE_NUMBER_PADDING_MAX,
                ),
                "number_start": Input.bool(True),
                "extension": Input.choice(SAVE_VIDEO_EXTENSIONS, DEFAULT_SAVE_VIDEO_EXTENSION),
                "quality": Input.int(
                    DEFAULT_SAVE_VIDEO_QUALITY,
                    SAVE_VIDEO_QUALITY_MIN,
                    SAVE_VIDEO_QUALITY_MAX,
                ),
                "save_info": Input.bool(True),
            },
            optional={
                "info": ("PIPELINE_INFO",),
                "audio": ("AUDIO",),
            },
        )

    RETURN_TYPES = ("VIDEO", "PIPELINE_INFO")
    RETURN_NAMES = ("video", "info")
    FUNCTION = "save_video"
    CATEGORY = CATEGORY_VIDEO

    def save_video(
        self,
        active: bool,
        images,
        video_config,
        folder: str,
        prefix: str,
        delimiter: str,
        number_padding: int,
        number_start: bool,
        extension: str,
        quality: int,
        save_info: bool,
        info=None,
        audio=None,
    ):
        options = SaveVideoOptions(
            folder=folder,
            prefix=prefix,
            delimiter=delimiter,
            number_padding=number_padding,
            number_start=number_start,
            extension=normalize_video_extension(extension),
            quality=quality,
            save_info=save_info,
        )

        video = create_video_from_images(images, video_config.fps, audio)
        info = pipeline_info(info, self._build_block(active, video_config, options))

        if active:
            tokens = TextTokens()
            saved = save_video_direct(
                video,
                build_filename_prefix(options.prefix, info.name, options.delimiter, tokens),
                options.extension,
                output_directory=folder_paths.get_output_directory(),
                subfolder=build_output_subfolder(options.folder, tokens),
                delimiter=options.delimiter,
                number_padding=options.number_padding,
                number_start=options.number_start,
                crf=video_crf(options.extension, options.quality),
            )
            if options.save_info:
                Path(f"{saved.full_path}.txt").write_text(info.text, encoding="utf-8")

            print(f"Video file saved to: {saved.full_path}")

        return (video, info)

    @staticmethod
    def _build_block(active: bool, video_config, options: SaveVideoOptions) -> str:
        return format_info(
            "Save Video",
            {
                "Active": active,
                "Extension": options.extension,
                "FPS": video_config.fps,
                "Quality": options.quality,
                "Crf": video_crf(options.extension, options.quality),
            },
        )


def video_crf(extension: str, quality: int) -> int | None:
    """The encoder's crf for a 1-10 quality, on the scale of the codec the container gets; None for GIF.

    The two codecs count differently - x264 runs 0-51 around 23, VP9 0-63 around 31 - so one number per
    container rather than one formula. GIF has no rate control and is left to its encoder.
    """
    scale = SAVE_VIDEO_CRF_RANGES.get(extension)
    if scale is None:
        return None

    best, worst = scale
    share = (SAVE_VIDEO_QUALITY_MAX - quality) / (SAVE_VIDEO_QUALITY_MAX - SAVE_VIDEO_QUALITY_MIN)
    return round(best + (worst - best) * share)


def create_video_from_images(images, fps: int | float, audio=None):
    try:
        from comfy_extras.nodes_video import CreateVideo
    except Exception as error:
        raise RuntimeError("ComfyUI CreateVideo node is not available.") from error

    result = CreateVideo.execute(images, float(fps), audio)
    return _first_result_value(result)


def save_video_direct(
    video,
    filename_prefix: str,
    extension: str,
    codec: str = "auto",
    output_directory: str | None = None,
    subfolder: str = "",
    delimiter: str = "_",
    number_padding: int = 4,
    number_start: bool = True,
    crf: int | None = None,
) -> SavedVideoFile:
    from comfy_api.latest import Types

    # The folder and the file name are separate on purpose: folder_paths.get_save_image_path() splits one
    # string on the last separator, which turns an empty file name into "the last folder is the file".
    output_directory = output_directory or folder_paths.get_output_directory()
    output_folder = os.path.join(output_directory, subfolder) if subfolder else output_directory

    if not folder_paths.is_within_directory(output_directory, output_folder):
        raise ValueError(f"Refusing to save a video outside of: {output_directory}")

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    container = _video_container(extension)
    file_extension = Types.VideoContainer.get_extension(container)
    file = next_available_filename(
        output_folder,
        filename_prefix,
        next_counter(output_folder, filename_prefix, delimiter, number_start),
        delimiter=delimiter,
        padding=number_padding,
        extension=file_extension,
        number_start=number_start,
    )
    full_path = os.path.join(output_folder, file)

    # metadata=None is intentional: calling native SaveVideo directly needs hidden
    # execution context. Direct save avoids extra_pnginfo/prompt assumptions.
    video.save_to(
        full_path,
        format=Types.VideoContainer(container),
        codec=codec,
        metadata=None,
        crf=crf,
    )

    return SavedVideoFile(file=file, subfolder=subfolder, full_path=full_path)


def _video_container(extension: str) -> str:
    extension = extension.lower().lstrip(".")
    if extension == "mp4":
        return "mp4"
    if extension == "webm":
        return "webm"
    if extension == "gif":
        return "gif"
    return "auto"


def _first_result_value(result):
    if isinstance(result, dict):
        values = result.get("result")
        if isinstance(values, tuple):
            return values[0]
        if isinstance(values, list):
            return values[0]
        return values

    values = getattr(result, "result", result)
    if callable(values):
        values = values()

    if isinstance(values, tuple):
        return values[0]

    if isinstance(values, list):
        return values[0]

    return values
