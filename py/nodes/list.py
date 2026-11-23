from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import comfy.samplers

from ...constants import *
from ..core.base import FlexibleOptionalInputType, any_type
from ..core.models import model_names
from .checkpoint import CustomCheckpointLoader
from .helpers import Input, InputGroup
from .loras import CustomLoraLoader


INDEX_INPUT = Input.int(DEFAULT_LIST_INDEX, LIST_INDEX_MIN, LIST_INDEX_MAX)
TEXT_LIST_INPUT = Input.string(multiline=True)


def _to_index(index: int | None) -> int:
    return max((index or 1) - 1, 0)


def _selected_names(kwargs: dict[str, Any]) -> list[str]:
    return [key for key, value in kwargs.items() if value is True]


def _select(values: list[Any], index: int | None, fallback: Any):
    resolved_index = _to_index(index)
    return values[resolved_index] if resolved_index < len(values) else fallback


def _non_empty_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _enabled_inputs(values: Iterable[str]) -> InputGroup:
    return {str(value): Input.bool(True) for value in values}


def _indexed_bool_inputs(values: Iterable[str]) -> InputGroup:
    return {"index": INDEX_INPUT, **_enabled_inputs(values)}


def _choice_result(index: int, kwargs: dict[str, Any], fallback: str = "") -> tuple[str, str]:
    value = _select(_selected_names(kwargs), index, fallback)
    return (value, value)


class TextListNode:
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_LIST

    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(index=INDEX_INPUT, list=TEXT_LIST_INPUT)


class IntList(TextListNode):
    RETURN_TYPES = ("INT",)

    def get_value(self, index: int, list: str):
        values = [int(line) for line in _non_empty_lines(list)]
        return (_select(values, index, 0),)


class FloatList(TextListNode):
    RETURN_TYPES = ("FLOAT",)

    def get_value(self, index: int, list: str):
        values = [round(float(line), 2) for line in _non_empty_lines(list)]
        return (_select(values, index, 0.0),)


class StringList(TextListNode):
    RETURN_TYPES = ("STRING",)

    def get_value(self, index: int, list: str):
        return (_select(_non_empty_lines(list), index, ""),)


class CheckpointList:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(**_indexed_bool_inputs(model_names(FOLDER_CHECKPOINTS)))

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "name")
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_LIST

    def get_value(self, index: int, **kwargs):
        name = _select(_selected_names(kwargs), index, "")
        if not name:
            return (None, None, None, "")

        return CustomCheckpointLoader().load_checkpoint(name)


class SamplerList:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(**_indexed_bool_inputs(comfy.samplers.KSampler.SAMPLERS))

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, "STRING")
    RETURN_NAMES = ("sampler", "name")
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_LIST

    def get_value(self, index: int, **kwargs):
        return _choice_result(index, kwargs)


class SchedulerList:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(**_indexed_bool_inputs(comfy.samplers.KSampler.SCHEDULERS))

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS, "STRING")
    RETURN_NAMES = ("scheduler", "name")
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_LIST

    def get_value(self, index: int, **kwargs):
        return _choice_result(index, kwargs)


class LorasList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "index": INDEX_INPUT,
            },
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "name")
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_LIST

    def get_value(self, model, clip, index: int, **kwargs):
        loras = [value for key, value in kwargs.items() if _is_enabled_lora(key, value)]
        selected = _select(loras, index, None)

        if selected is None:
            return (model, clip, "None")

        return CustomLoraLoader().load_lora(
            model,
            clip,
            selected["lora"],
            selected["strength"],
        )


def _is_enabled_lora(key: str, value: Any) -> bool:
    if not key.upper().startswith("LORA_"):
        return False

    if not isinstance(value, dict):
        return False

    if not all(field in value for field in ("on", "lora", "strength")):
        return False

    return bool(value["on"]) and round(value["strength"], 2) != 0


class StaticChoiceList:
    VALUES: Iterable[str] = ()
    RETURN_KEY = "value"
    RETURN_TYPES: tuple[Any, str]
    RETURN_NAMES: tuple[str, str]

    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(**_indexed_bool_inputs(cls.VALUES))

    FUNCTION = "get_value"
    CATEGORY = CATEGORY_LIST

    def get_value(self, index: int, **kwargs):
        return _choice_result(index, kwargs)


class DimensionsList(StaticChoiceList):
    VALUES = DIMENSIONS
    RETURN_TYPES = (DIMENSIONS, "STRING")
    RETURN_NAMES = ("dimension", "name")


class VideoDimensionsList(StaticChoiceList):
    VALUES = VIDEO_DIMENSIONS
    RETURN_TYPES = (VIDEO_DIMENSIONS, "STRING")
    RETURN_NAMES = ("video_dimensions", "name")


class UpscaleLatentModeList(StaticChoiceList):
    VALUES = UPSCALE_LATENT_MODES
    RETURN_TYPES = (UPSCALE_LATENT_MODES, "STRING")
    RETURN_NAMES = ("upscale_latent_mode", "name")


class ImageResizeMethodList(StaticChoiceList):
    VALUES = IMAGE_RESIZE_METHODS
    RETURN_TYPES = (IMAGE_RESIZE_METHODS, "STRING")
    RETURN_NAMES = ("image_resize_method", "name")


class PromptSeparatorList(StaticChoiceList):
    VALUES = PROMPT_SEPARATORS
    RETURN_TYPES = (PROMPT_SEPARATORS, "STRING")
    RETURN_NAMES = ("prompt_separator", "name")


class PositiveStylePositionList(StaticChoiceList):
    VALUES = POSITIVE_STYLE_POSITIONS
    RETURN_TYPES = (POSITIVE_STYLE_POSITIONS, "STRING")
    RETURN_NAMES = ("positive_style_position", "name")
