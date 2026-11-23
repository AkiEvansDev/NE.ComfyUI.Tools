from __future__ import annotations

from ...constants import CATEGORY_ROOT
from .helpers import Input


def format_active_group(raw_group: str) -> str | None:
    group = raw_group.strip()
    if not group:
        return None

    return f"[{group.split('. ')[-1]}]"


def format_active_groups(raw_groups: str) -> str:
    groups = [format_active_group(group) for group in raw_groups.split(",")]
    return ", ".join(group for group in groups if group)


class GroupsMuter:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(groups=Input.string())

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("active_groups",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_ROOT

    def execute(self, groups: str):
        return (format_active_groups(groups),)
