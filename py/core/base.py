from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import folder_paths

from ...constants import *
from .models import load_upscale_model


def round_float(value: float) -> float:
    return round(value, ROUND_DIGITS)


def append_info(info: str | None, block: str) -> str:
    if info and info.strip():
        return f"{info.rstrip()}\n\n{block}"

    return block


@dataclass(slots=True)
class PipelineInfo:
    """The log and the output name travelling together as one link.

    They are always produced and consumed at the same points, so two parallel wires running the length
    of a graph bought nothing but clutter.
    """

    text: str = ""
    name: str = ""

    def appended(self, block: str, name: str | None = None) -> "PipelineInfo":
        return PipelineInfo(text=append_info(self.text, block), name=self.name if name is None else name)

    def get_copy(self) -> "PipelineInfo":
        return replace(self)

    # DisplayAny falls back to str() for anything it cannot serialise, so this is what the node shows.
    def __str__(self) -> str:
        return self.text


@dataclass(slots=True)
class PromptConfig:
    positive: str = ""
    posture: str = ""
    background: str = ""
    negative: str = ""
    positive_style: str = ""
    negative_style: str = ""
    separator: str = DEFAULT_PROMPT_SEPARATOR
    positive_style_position: str = DEFAULT_POSITIVE_STYLE_POSITION
    # Carried rather than applied and forgotten: a step that builds a prompt of its own out of these parts
    # has to weight its own text the same way, and cannot tell from the text whether the setting was on.
    process_tag_weights: bool = DEFAULT_PROCESS_TAG_WEIGHTS

    def get_copy(self) -> "PromptConfig":
        return replace(self)


@dataclass(slots=True)
class ModelConfig:
    """A model together with everything derived from it that a sampling step needs.

    The model, its clip and the prompts encoded with that clip always travel together, and wiring them
    as four separate links to every step is what made swapping in a hires variant so tedious.
    """

    model: Any = None
    clip: Any = None
    positive: Any = None
    negative: Any = None
    positive_text: str = ""
    negative_text: str = ""
    # The parts the encoded conditioning was built from. A step that conditions a crop rather than the whole
    # frame needs to pick a subset of them, and the joined text cannot be taken apart again.
    prompt: PromptConfig | None = None
    model_name: str = ""
    loras_name: str = ""

    def get_copy(self) -> "ModelConfig":
        return replace(self, prompt=self.prompt.get_copy() if self.prompt is not None else None)

    def __str__(self) -> str:
        return f"(positive={self.positive_text}, negative={self.negative_text})"


@dataclass(slots=True)
class VideoModelConfig:
    """The video counterpart of ModelConfig: both experts, the encoder and vae they share, and the prompts.

    A Wan step used to take five separate links for this, and the Wan 2.2 pair added two more. They are
    chosen together and never mean anything apart, so they travel as one.
    """

    model: Any = None
    model_low: Any = None
    clip: Any = None
    vae: Any = None
    clip_vision: Any = None
    positive: Any = None
    negative: Any = None
    positive_text: str = ""
    negative_text: str = ""
    model_name: str = ""
    loras_name: str = ""
    boundary: float = DEFAULT_WAN_BOUNDARY

    def __post_init__(self) -> None:
        self.boundary = round_float(
            SamplerConfig._validate_range(
                self.boundary,
                WAN_BOUNDARY_MIN,
                WAN_BOUNDARY_MAX,
                "Boundary must be between 0.0 and 1.0!",
            )
        )

    def get_copy(self) -> "VideoModelConfig":
        return replace(self)

    def __str__(self) -> str:
        return f"(positive={self.positive_text}, negative={self.negative_text})"


@dataclass(slots=True)
class SamplerConfig:
    seed: int = 0
    sampler: str = DEFAULT_SAMPLER
    scheduler: str = DEFAULT_SCHEDULER
    steps: int = DEFAULT_STEPS
    cfg: float = DEFAULT_CFG
    denoise: float = DEFAULT_DENOISE

    def __post_init__(self) -> None:
        self.seed = self._validate_min(self.seed, 0, "Seed must be non-negative!")
        self.steps = self._validate_min(self.steps, 1, "Steps must be greater than 0!")
        self.cfg = self._validate_min(self.cfg, 0.0, "CFG must be greater than 0!", strict=True)
        self.denoise = self._validate_range(
            self.denoise,
            DENOISE_MIN,
            DENOISE_MAX,
            "Denoise must be between 0.0 and 1.0!",
        )

    def get_copy(self) -> "SamplerConfig":
        return replace(self)

    def __str__(self) -> str:
        return (
            f"(seed={self.seed}, sampler={self.sampler}, scheduler={self.scheduler}, "
            f"steps={self.steps}, cfg={self.cfg}, denoise={self.denoise})"
        )

    @staticmethod
    def _validate_min(value: Any, minimum: float, message: str, strict: bool = False) -> Any:
        if strict:
            if value > minimum:
                return value
        elif value >= minimum:
            return value

        raise ValueError(message)

    @staticmethod
    def _validate_range(value: Any, minimum: float, maximum: float, message: str) -> Any:
        if minimum <= value <= maximum:
            return value

        raise ValueError(message)


@dataclass(slots=True)
class HiresConfig(SamplerConfig):
    steps: int = DEFAULT_HIRES_STEPS
    cfg: float = DEFAULT_HIRES_CFG
    denoise: float = DEFAULT_HIRES_DENOISE
    scale: float = DEFAULT_HIRES_SCALE
    upscale_mode: str = DEFAULT_UPSCALE_LATENT_MODE

    def __post_init__(self) -> None:
        SamplerConfig.__post_init__(self)
        self.scale = self._validate_min(self.scale, 0.0, "Scale must be greater than 0!", strict=True)

    def get_copy(self) -> "HiresConfig":
        return replace(self)

    def __str__(self) -> str:
        return f"(scale={self.scale}, upscale_mode={self.upscale_mode}) + {super().__str__()}"


@dataclass(slots=True)
class ImageUpscaleConfig:
    model_name: str = ""
    upscale_method: str = DEFAULT_IMAGE_RESIZE_METHOD
    resize: float = DEFAULT_IMAGE_UPSCALE_RESIZE
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.resize = round_float(SamplerConfig._validate_min(
            self.resize,
            IMAGE_UPSCALE_RESIZE_MIN,
            "Resize must be greater than or equal to 0.05!",
            strict=False,
        ))

    @property
    def model(self) -> Any:
        """The weights, loaded on first use.

        Building a config has to stay free, otherwise a step that is switched off still pays a disk read
        and every caller needs a second, config-less path for its info block.
        """
        if self._model is None:
            self._model = load_upscale_model(self.model_name)

        return self._model

    def get_copy(self) -> "ImageUpscaleConfig":
        copy = replace(self)
        # replace() skips init=False fields, and re-reading the weights from disk for a copy is pure waste.
        copy._model = self._model
        return copy

    def __str__(self) -> str:
        return f"(model={self.model_name}, upscale_method={self.upscale_method}, resize={self.resize})"


VIDEO_UPSCALE_RANGES = (
    ("temporal_smoothing", VIDEO_TEMPORAL_SMOOTHING_MIN, VIDEO_TEMPORAL_SMOOTHING_MAX),
    ("temporal_sharpen", VIDEO_TEMPORAL_SHARPEN_MIN, VIDEO_TEMPORAL_SHARPEN_MAX),
    ("motion_sensitivity", VIDEO_MOTION_SENSITIVITY_MIN, VIDEO_MOTION_SENSITIVITY_MAX),
    ("edge_preservation", VIDEO_EDGE_PRESERVATION_MIN, VIDEO_EDGE_PRESERVATION_MAX),
    ("sharpen", VIDEO_SHARPEN_MIN, VIDEO_SHARPEN_MAX),
)


def _range_message(name: str, minimum: float, maximum: float) -> str:
    return f"{name.replace('_', ' ').capitalize()} must be between {minimum} and {maximum}!"


@dataclass(slots=True)
class VideoUpscaleConfig:
    image_upscale: ImageUpscaleConfig
    batch_size: int = DEFAULT_VIDEO_UPSCALE_BATCH_SIZE
    temporal_smoothing_mode: str = DEFAULT_VIDEO_TEMPORAL_SMOOTHING_MODE
    temporal_smoothing: float = DEFAULT_VIDEO_TEMPORAL_SMOOTHING
    temporal_sharpen: float = DEFAULT_VIDEO_TEMPORAL_SHARPEN
    motion_sensitivity: float = DEFAULT_VIDEO_MOTION_SENSITIVITY
    edge_aware_smoothing: bool = DEFAULT_VIDEO_EDGE_AWARE_SMOOTHING
    edge_preservation: float = DEFAULT_VIDEO_EDGE_PRESERVATION
    sharpen: float = DEFAULT_VIDEO_SHARPEN

    def __post_init__(self) -> None:
        for name, minimum, maximum in VIDEO_UPSCALE_RANGES:
            value = round_float(getattr(self, name))
            setattr(self, name, SamplerConfig._validate_range(value, minimum, maximum, _range_message(name, minimum, maximum)))

        if self.temporal_smoothing_mode not in VIDEO_TEMPORAL_SMOOTHING_MODES:
            raise ValueError(f"Temporal smoothing mode must be one of: {', '.join(VIDEO_TEMPORAL_SMOOTHING_MODES)}!")

        self.batch_size = int(SamplerConfig._validate_range(
            int(self.batch_size),
            VIDEO_UPSCALE_BATCH_SIZE_MIN,
            VIDEO_UPSCALE_BATCH_SIZE_MAX,
            _range_message("batch_size", VIDEO_UPSCALE_BATCH_SIZE_MIN, VIDEO_UPSCALE_BATCH_SIZE_MAX),
        ))
        self.edge_aware_smoothing = bool(self.edge_aware_smoothing)

    @property
    def model(self):
        return self.image_upscale.model

    @property
    def model_name(self) -> str:
        return self.image_upscale.model_name

    @property
    def upscale_method(self) -> str:
        return self.image_upscale.upscale_method

    @property
    def resize(self) -> float:
        return self.image_upscale.resize

    def get_copy(self) -> "VideoUpscaleConfig":
        return replace(self, image_upscale=self.image_upscale.get_copy())

    def __str__(self) -> str:
        return (
            f"(image_upscale={self.image_upscale}, temporal_smoothing={self.temporal_smoothing}, "
            f"temporal_smoothing_mode={self.temporal_smoothing_mode}, "
            f"motion_sensitivity={self.motion_sensitivity}, batch_size={self.batch_size}, sharpen={self.sharpen}, "
            f"edge_aware_smoothing={self.edge_aware_smoothing}, edge_preservation={self.edge_preservation}, "
            f"temporal_sharpen={self.temporal_sharpen})"
        )


@dataclass(slots=True)
class VideoConfig:
    dimensions: str = DEFAULT_VIDEO_DIMENSIONS
    width: int = 640
    height: int = 640
    frames: int = DEFAULT_VIDEO_FRAMES
    fps: int = DEFAULT_VIDEO_FPS
    shift: float = DEFAULT_WAN_SHIFT

    def __post_init__(self) -> None:
        self.frames = SamplerConfig._validate_min(self.frames, 1, "Frames must be greater than 0!")
        self.fps = SamplerConfig._validate_min(self.fps, 1, "FPS must be greater than 0!")
        self.width = SamplerConfig._validate_min(self.width, 1, "Width must be greater than 0!")
        self.height = SamplerConfig._validate_min(self.height, 1, "Height must be greater than 0!")
        self.shift = SamplerConfig._validate_min(self.shift, 0.0, "Shift must be greater than or equal to 0!")

    @property
    def duration(self) -> float:
        return self.frames / self.fps

    def get_copy(self) -> "VideoConfig":
        return replace(self)

    def __str__(self) -> str:
        return (
            f"(dimensions={self.dimensions}, width={self.width}, height={self.height}, "
            f"frames={self.frames}, fps={self.fps}, shift={self.shift})"
        )


@dataclass(frozen=True)
class XYRangeConfig:
    x: int
    y: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int

    @property
    def x_count(self) -> int:
        return abs(self.x_end - self.x_start) + 1

    @property
    def y_count(self) -> int:
        return abs(self.y_end - self.y_start) + 1

    @property
    def total(self) -> int:
        return self.x_count * self.y_count

    @property
    def first_cell(self) -> bool:
        return self.x == self.x_start and self.y == self.y_start

    @property
    def signature(self) -> tuple[int, int, int, int]:
        return (self.x_start, self.x_end, self.y_start, self.y_end)

@dataclass(frozen=True)
class XYZRangeConfig(XYRangeConfig):
    z: int
    z_start: int
    z_end: int

    @property
    def z_count(self) -> int:
        return abs(self.z_end - self.z_start) + 1

    @property
    def total(self) -> int:
        return self.x_count * self.y_count * self.z_count

    @property
    def first_cell(self) -> bool:
        return self.x == self.x_start and self.y == self.y_start and self.z == self.z_start

    @property
    def signature(self) -> tuple[int, int, int, int, int, int]:
        return (self.x_start, self.x_end, self.y_start, self.y_end, self.z_start, self.z_end)


class AnyType(str):
    def __ne__(self, value: object) -> bool:
        return False


class FlexibleOptionalInputType(dict):
    def __init__(self, input_type: Any):
        super().__init__()
        self.input_type = input_type

    def __getitem__(self, key: str) -> tuple[Any]:
        return (self.input_type,)

    def __contains__(self, key: object) -> bool:
        return True


any_type = AnyType("*")


def is_none(value: Any) -> bool:
    if isinstance(value, str):
        return is_blank(value)

    if isinstance(value, dict) and "model" in value and "clip" in value:
        return not value or all(item is None for item in value.values())

    return value is None


def is_blank(value: str | None) -> bool:
    return not (value and value.strip())


def is_not_blank(value: str | None) -> bool:
    return bool(value and value.strip())


def clean_string(value: str | None) -> str | None:
    if is_not_blank(value):
        return " ".join(value.strip().split())

    return value


def extract_filename(filepath: str) -> str:
    path = Path(filepath)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    model_suffixes = {
        ".safetensors",
        ".ckpt",
        ".pt",
        ".pth",
        ".bin",
        ".onnx",
        ".gguf",
        ".ggml",
        ".zip",
    }

    if suffixes and suffixes[-1] in model_suffixes:
        return path.name[: -len(path.suffix)]

    return path.name


def filter_non_empty_strings(strings: Iterable[str]) -> list[str]:
    return [value for value in strings if is_not_blank(value)]


def add_folder_path(folder_name: str, extensions_to_register: list[str]) -> None:
    path = os.path.join(folder_paths.models_dir, folder_name)
    folders, extensions = folder_paths.folder_names_and_paths.get(folder_name, ([], set()))

    if path not in folders:
        folders.append(path)

    if isinstance(extensions, set):
        extensions.update(extensions_to_register)
    elif isinstance(extensions, list):
        extensions.extend(extension for extension in extensions_to_register if extension not in extensions)
    else:
        raise TypeError(
            f"Failed to register models/{folder_name} folder. "
            f"Expected set or list of extensions, got {type(extensions).__name__}: {extensions}"
        )

    folder_paths.folder_names_and_paths[folder_name] = (folders, extensions)


def get_path_by_filename(file_path: str, folder: str) -> str | None:
    available_paths = folder_paths.get_filename_list(folder)

    if file_path in available_paths:
        return file_path

    path_without_extension = os.path.splitext(file_path)[0]
    available_paths_without_extension = [os.path.splitext(path)[0] for path in available_paths]

    if file_path in available_paths_without_extension:
        return available_paths[available_paths_without_extension.index(file_path)]

    if path_without_extension in available_paths_without_extension:
        return available_paths[available_paths_without_extension.index(path_without_extension)]

    return None