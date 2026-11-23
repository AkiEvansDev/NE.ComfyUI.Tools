from __future__ import annotations

from ...constants import CATEGORY_STRING
from ..core.base import FlexibleOptionalInputType, any_type, is_not_blank
from .convert import to_string
from .helpers import Input


def _decode_newlines(value: str) -> str:
    return value.replace("\\n", "\n") if is_not_blank(value) else value


def _single_line(default: str = ""):
    return Input.string(default=default)


class StringLength:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(value=_single_line())

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_STRING

    def get_value(self, value: str):
        return (len(value),)


class StringConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            a=_single_line(),
            separator=_single_line(),
            b=_single_line(),
        )

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_STRING

    def get_value(self, a: str, separator: str, b: str):
        a = _decode_newlines(a)
        separator = _decode_newlines(separator)
        b = _decode_newlines(b)

        return (f"{a}{separator}{b}" if is_not_blank(b) else a,)


class StringReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            value=_single_line(),
            find=_single_line(),
            replace=_single_line(),
        )

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_STRING

    def get_value(self, value: str, find: str, replace: str):
        return (value.replace(find, replace),)


class StringEquals:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(a=_single_line(), b=_single_line())

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_STRING

    def get_value(self, a: str, b: str):
        return (a == b,)


class ToStringConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": _single_line(),
                "separator": _single_line(),
            },
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_STRING

    def get_value(self, template: str, separator: str, **kwargs):
        result = _decode_newlines(template) if is_not_blank(template) else ""
        separator = _decode_newlines(separator)
        extra_parts: list[str] = []

        for key, value in kwargs.items():
            string_value = to_string(value)
            placeholder = f"{{{key}}}"

            if placeholder in result:
                result = result.replace(placeholder, string_value)
            else:
                extra_parts.append(string_value)

        if not extra_parts:
            return (result,)

        if is_not_blank(result):
            return (separator.join([result, *extra_parts]),)

        return (separator.join(extra_parts),)
