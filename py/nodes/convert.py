from __future__ import annotations

from ...constants import CATEGORY_CONVERT
from ..core.base import any_type
from .helpers import Input


def to_string(value) -> str:
    if value is None:
        return "None"

    try:
        return str(value)
    except Exception:
        return "error"


class IntToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(value=Input.force("INT"))

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_CONVERT

    def get_value(self, value: int):
        return (float(value),)


class FloatToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(value=Input.force("FLOAT"))

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_CONVERT

    def get_value(self, value: float):
        return (int(value),)


class ToString:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(source=(any_type, {}))

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_CONVERT

    def get_value(self, source=None):
        return (to_string(source),)
