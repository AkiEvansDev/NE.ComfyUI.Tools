from __future__ import annotations

from typing import Any, TypedDict

from nodes import LoraLoader, LoraLoaderModelOnly

from ...constants import *
from ..core.base import FlexibleOptionalInputType, any_type, extract_filename, get_path_by_filename
from ..core.models import expert_pair, model_names
from .helpers import Input


class LoraInput(TypedDict):
    on: bool
    lora: str
    strength: float


LORA_INPUT_FIELDS = ("on", "lora", "strength")


def _lora_strength_input():
    return Input.float(
        DEFAULT_LORA_STRENGTH,
        LORA_STRENGTH_MIN,
        LORA_STRENGTH_MAX,
        LORA_STRENGTH_STEP,
    )


def _format_lora_name(lora: str, strength: float) -> str:
    return f"{extract_filename(lora)}:{strength:.2f}"


def _resolve_lora_path(lora: str) -> str:
    lora_path = get_path_by_filename(lora, FOLDER_LORAS)
    if lora_path is None:
        raise ValueError(f"LoRA not found: {lora}")

    return lora_path


def _is_lora_input(key: str, value: Any) -> bool:
    return (
        key.upper().startswith("LORA_")
        and isinstance(value, dict)
        and all(field in value for field in LORA_INPUT_FIELDS)
    )


def _normalize_strength(value: float) -> float:
    return round(float(value), 2)


def _load_lora(model, clip, lora: str, strength: float):
    strength = _normalize_strength(strength)
    model, clip = LoraLoader().load_lora(
        model,
        clip,
        _resolve_lora_path(lora),
        strength,
        strength,
    )

    return model, clip, _format_lora_name(lora, strength)


class LorasLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            optional=FlexibleOptionalInputType(any_type),
        )

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "names")
    FUNCTION = "load_lora"
    CATEGORY = CATEGORY_LORA

    def load_lora(self, model, clip, **kwargs):
        names: list[str] = []

        for key, value in kwargs.items():
            if not _is_lora_input(key, value):
                continue

            if not value["on"]:
                continue

            strength = _normalize_strength(value["strength"])
            if strength == 0:
                continue

            model, clip, name = _load_lora(model, clip, value["lora"], strength)
            names.append(f"- {name}")

        return (model, clip, "\n".join(names))


class WanExpertLoras:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "clip": ("CLIP",),
            },
            optional=FlexibleOptionalInputType(any_type),
        )

    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model_high", "model_low", "clip", "names")
    FUNCTION = "load_lora"
    CATEGORY = CATEGORY_LORA

    def load_lora(self, model_high, model_low, clip, **kwargs):
        names: list[str] = []

        for key, value in kwargs.items():
            if not _is_lora_input(key, value):
                continue

            if not value["on"]:
                continue

            strength = _normalize_strength(value["strength"])
            if strength == 0:
                continue

            high, low = expert_pair(value["lora"], FOLDER_LORAS)
            # The clip side is patched once, on the high pass: both files carry the same text-encoder
            # weights when they carry any at all, and applying them twice would double the strength.
            model_high, clip, name = _load_lora(model_high, clip, high, strength)
            names.append(f"- {name}")

            model_low, = LoraLoaderModelOnly().load_lora_model_only(model_low, _resolve_lora_path(low), strength)
            if low != high:
                names.append(f"- {_format_lora_name(low, strength)}")

        return (model_high, model_low, clip, "\n".join(names))


class CustomLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            model=("MODEL",),
            clip=("CLIP",),
            lora=Input.choice(model_names(FOLDER_LORAS)),
            strength=_lora_strength_input(),
        )

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "name")
    FUNCTION = "load_lora"
    CATEGORY = CATEGORY_LORA

    def load_lora(self, model, clip, lora: str, strength: float):
        return _load_lora(model, clip, lora, strength)
