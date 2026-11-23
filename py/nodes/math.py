from __future__ import annotations

from collections.abc import Callable

from ...constants import *
from .helpers import Input


MathFn = Callable[[float, float], float]
CompareFn = Callable[[float, float], bool]


_MATH: dict[str, MathFn] = {
    "a + b": lambda a, b: a + b,
    "a - b": lambda a, b: a - b,
    "a * b": lambda a, b: a * b,
    "a / b": lambda a, b: a / b if b != 0 else 1,
}

_COMPARE: dict[str, CompareFn] = {
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a == b": lambda a, b: a == b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
}


def _int_input(default: int = 0):
    return Input.int(default, INT_MIN, INT_MAX)


def _float_input(default: float = 0.0):
    return Input.float(default, FLOAT_MIN, FLOAT_MAX, FLOAT_STEP)


def _math_result(a: float, b: float, action: str) -> float:
    return _MATH.get(action, lambda x, _y: x)(a, b)


def _compare_result(a: float, b: float, action: str) -> bool:
    return _COMPARE.get(action, lambda _x, _y: False)(a, b)


class MathInt:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            a=_int_input(),
            b=_int_input(),
            action=Input.choice(MATH_ACTIONS, "a + b"),
        )

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_MATH

    def get_value(self, a: int, b: int, action: str):
        return (int(_math_result(a, b, action)),)


class MathFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            a=_float_input(),
            b=_float_input(),
            action=Input.choice(MATH_ACTIONS, "a + b"),
        )

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_MATH

    def get_value(self, a: float, b: float, action: str):
        return (round(_math_result(round(a, 2), round(b, 2), action), 4),)


class CompareInt:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            a=_int_input(),
            b=_int_input(),
            action=Input.choice(COMPARE_ACTIONS, "a < b"),
        )

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_MATH

    def get_value(self, a: int, b: int, action: str):
        return (_compare_result(a, b, action),)


class CompareFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            a=_float_input(),
            b=_float_input(),
            action=Input.choice(COMPARE_ACTIONS, "a < b"),
        )

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_MATH

    def get_value(self, a: float, b: float, action: str):
        return (_compare_result(round(a, 2), round(b, 2), action),)
