from __future__ import annotations

from ...constants import CATEGORY_SWITCH, FLOAT_MAX, FLOAT_MIN, FLOAT_STEP, INT_MAX, INT_MIN
from ..core.base import FlexibleOptionalInputType, any_type, is_none
from .helpers import Input


def _select(a, b, condition: bool):
    return a if condition else b


class AnySwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("result",)
    FUNCTION = "switch"
    CATEGORY = CATEGORY_SWITCH

    def switch(self, **kwargs):
        any_value = None
        for key, value in kwargs.items():
            if key.isdigit() and not is_none(value):
                any_value = value
                break
        return (any_value,)


class AnyTypeSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            a=(any_type, {}),
            b=(any_type, {}),
            condition=Input.bool(True),
        )

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("result",)
    FUNCTION = "switch"
    CATEGORY = CATEGORY_SWITCH

    def switch(self, a, b, condition: bool):
        return (_select(a, b, condition),)


class IntSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            a=Input.int(0, INT_MIN, INT_MAX),
            b=Input.int(0, INT_MIN, INT_MAX),
            condition=Input.bool(True),
        )

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "switch"
    CATEGORY = CATEGORY_SWITCH

    def switch(self, a: int, b: int, condition: bool):
        return (_select(a, b, condition),)


class FloatSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            a=Input.float(0.0, FLOAT_MIN, FLOAT_MAX, FLOAT_STEP),
            b=Input.float(0.0, FLOAT_MIN, FLOAT_MAX, FLOAT_STEP),
            condition=Input.bool(True),
        )

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "switch"
    CATEGORY = CATEGORY_SWITCH

    def switch(self, a: float, b: float, condition: bool):
        return (round(_select(a, b, condition), 2),)


class StringSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            a=Input.string(),
            b=Input.string(),
            condition=Input.bool(True),
        )

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "switch"
    CATEGORY = CATEGORY_SWITCH

    def switch(self, a: str, b: str, condition: bool):
        return (_select(a, b, condition),)
