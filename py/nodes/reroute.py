from __future__ import annotations

from ...constants import CATEGORY_ROOT
from ..core.base import any_type
from .helpers import Input


class Reroute:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(value=(any_type, {}))

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_ROOT

    def execute(self, value):
        return (value,)
