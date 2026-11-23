from __future__ import annotations

from server import PromptServer

from ...constants import CATEGORY_RANGE, INT_MAX, INT_MIN, RANGE_FEEDBACK_EVENT, XY_RANGE_FEEDBACK_EVENT, XYZ_RANGE_FEEDBACK_EVENT
from ..core.base import XYRangeConfig, XYZRangeConfig
from ..server.state import reset_registry
from .helpers import Input


def _range_int(default: int = 0):
    return Input.int(default, INT_MIN, INT_MAX)


def _normalize_bounds(start: int, end: int) -> tuple[int, int]:
    return (end, start) if start > end else (start, end)


def _ensure_registry(unique_id: str | None) -> None:
    if unique_id:
        reset_registry.setdefault(unique_id, False)


def _pop_reset(unique_id: str | None) -> bool:
    if not unique_id or not reset_registry.get(unique_id):
        return False

    reset_registry[unique_id] = False
    return True


def _send_range_feedback(unique_id: str | None, payload: dict) -> None:
    PromptServer.instance.send_sync(RANGE_FEEDBACK_EVENT, {"node_id": unique_id, **payload})


def _send_xy_range_feedback(unique_id: str | None, payload: dict) -> None:
    PromptServer.instance.send_sync(XY_RANGE_FEEDBACK_EVENT, {"node_id": unique_id, **payload})


def _send_xyz_range_feedback(unique_id: str | None, payload: dict) -> None:
    PromptServer.instance.send_sync(XYZ_RANGE_FEEDBACK_EVENT, {"node_id": unique_id, **payload})


def _parse_int_lines(value: str) -> list[int]:
    return [int(line.strip()) for line in value.splitlines() if line.strip()]


class StatefulRangeNode:
    def __init__(self):
        self._unique_id: str | None = None

    def __del__(self):
        if self._unique_id is not None:
            reset_registry.pop(self._unique_id, None)

    def _bind(self, unique_id: str | None) -> bool:
        self._unique_id = unique_id
        _ensure_registry(unique_id)
        return _pop_reset(unique_id)


class Range(StatefulRangeNode):
    def __init__(self):
        super().__init__()
        self._current = 0

    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "current": _range_int(0),
                "start": _range_int(1),
                "end": _range_int(2),
            },
            hidden={"unique_id": "UNIQUE_ID"},
        )

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("current",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_RANGE

    def get_value(self, current: int, start: int, end: int, unique_id: str | None = None):
        freeze = self._bind(unique_id)
        if freeze:
            self._current = current

        start, end = _normalize_bounds(start, end)
        current = self._current

        if current < start or current > end:
            current = start
        elif not freeze:
            current = start if current >= end else current + 1

        self._current = current
        _send_range_feedback(unique_id, {"current": current, "start": start, "end": end})

        return (current,)

    @classmethod
    def IS_CHANGED(cls, current, start, end, unique_id):
        return float("NaN")


class XYRange(StatefulRangeNode):
    def __init__(self):
        super().__init__()
        self._x = 0
        self._y = 0

    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "x": _range_int(0),
                "y": _range_int(0),
                "x_start": _range_int(1),
                "x_end": _range_int(2),
                "y_start": _range_int(1),
                "y_end": _range_int(2),
            },
            hidden={"unique_id": "UNIQUE_ID"},
        )

    RETURN_TYPES = ("XY_RANGE_CONFIG", "INT", "INT")
    RETURN_NAMES = ("config", "x", "y")
    FUNCTION = "execute"
    CATEGORY = CATEGORY_RANGE

    def execute(
        self,
        x: int,
        y: int,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
        unique_id: str | None = None,
    ):
        freeze = self._bind(unique_id)
        if freeze:
            self._x = x
            self._y = y

        x_start, x_end = _normalize_bounds(x_start, x_end)
        y_start, y_end = _normalize_bounds(y_start, y_end)

        x = self._x
        y = self._y

        if x < x_start or x > x_end:
            x = x_start

        if y < y_start or y > y_end:
            y = y_start
        elif not freeze:
            y += 1
            if y > y_end:
                y = y_start
                x = x_start if x >= x_end else x + 1

        self._x = x
        self._y = y

        payload = {
            "x": x,
            "y": y,
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end,
        }
        _send_xy_range_feedback(unique_id, payload)

        config = XYRangeConfig(
            x=x,
            y=y,
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end,
        )

        return (config, x, y)

    @classmethod
    def IS_CHANGED(cls, x, y, x_start, x_end, y_start, y_end, unique_id):
        return float("NaN")


class XYZRange(StatefulRangeNode):
    def __init__(self):
        super().__init__()
        self._x = 0
        self._y = 0
        self._z = 0

    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "x": _range_int(0),
                "y": _range_int(0),
                "z": _range_int(0),
                "x_start": _range_int(1),
                "x_end": _range_int(2),
                "y_start": _range_int(1),
                "y_end": _range_int(2),
                "z_start": _range_int(1),
                "z_end": _range_int(2),
            },
            hidden={"unique_id": "UNIQUE_ID"},
        )

    RETURN_TYPES = ("XYZ_RANGE_CONFIG", "INT", "INT", "INT")
    RETURN_NAMES = ("config", "x", "y", "z")
    FUNCTION = "execute"
    CATEGORY = CATEGORY_RANGE

    def execute(
        self,
        x: int,
        y: int,
        z: int,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
        z_start: int,
        z_end: int,
        unique_id: str | None = None,
    ):
        freeze = self._bind(unique_id)
        if freeze:
            self._x = x
            self._y = y
            self._z = z

        x_start, x_end = _normalize_bounds(x_start, x_end)
        y_start, y_end = _normalize_bounds(y_start, y_end)
        z_start, z_end = _normalize_bounds(z_start, z_end)

        x = self._x
        y = self._y
        z = self._z

        if x < x_start or x > x_end:
            x = x_start

        if y < y_start or y > y_end:
            y = y_start

        if z < z_start or z > z_end:
            z = z_start
        elif not freeze:
            z += 1
            if z > z_end:
                z = z_start
                y += 1
                if y > y_end:
                    y = y_start
                    x = x_start if x >= x_end else x + 1

        self._x = x
        self._y = y
        self._z = z

        payload = {
            "x": x,
            "y": y,
            "z": z,
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end,
            "z_start": z_start,
            "z_end": z_end,
        }
        _send_xyz_range_feedback(unique_id, payload)

        config = XYZRangeConfig(
            x=x,
            y=y,
            z=z,
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end,
            z_start=z_start,
            z_end=z_end,
        )

        return (config, x, y, z)

    @classmethod
    def IS_CHANGED(cls, x, y, z, x_start, x_end, y_start, y_end, z_start, z_end, unique_id):
        return float("NaN")


class RangeList(StatefulRangeNode):
    def __init__(self):
        super().__init__()
        self._current = 0

    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "current": _range_int(0),
                "list": Input.string(multiline=True),
            },
            hidden={"unique_id": "UNIQUE_ID"},
        )

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("current",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_RANGE

    def get_value(self, current: int, list: str, unique_id: str | None = None):
        freeze = self._bind(unique_id)
        if freeze:
            self._current = current

        values = _parse_int_lines(list)
        if not values:
            self._current = current
            return (current,)

        current = self._current
        index = values.index(current) if current in values else -1

        if index < 0:
            index = 0
        elif not freeze:
            index = 0 if index >= len(values) - 1 else index + 1

        current = values[index]
        self._current = current

        _send_range_feedback(unique_id, {"current": current})

        return (current,)

    @classmethod
    def IS_CHANGED(cls, current, list, unique_id):
        return float("NaN")
