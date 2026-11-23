from __future__ import annotations

from ...constants import *
from .helpers import Input


class Bool:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(value=Input.bool(True))

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_PRIMITIVES

    def get_value(self, value: bool):
        return (value,)


class Int:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(value=Input.int(0, INT_MIN, INT_MAX))

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_PRIMITIVES

    def get_value(self, value: int):
        return (value,)


class Float:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(value=Input.float(0.0, FLOAT_MIN, FLOAT_MAX, FLOAT_STEP))

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_PRIMITIVES

    def get_value(self, value: float):
        return (round(value, 2),)


class String:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(value=Input.string())

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_PRIMITIVES

    def get_value(self, value: str):
        return (value,)


class Text:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(value=Input.string(multiline=True))

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_PRIMITIVES

    def get_value(self, value: str):
        return (value,)
