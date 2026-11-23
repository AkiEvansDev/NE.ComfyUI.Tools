from __future__ import annotations

from typing import Any, TypeAlias

from ....constants import (
    CFG_MAX,
    CFG_MIN,
    CFG_STEP,
    DEFAULT_CFG,
    DEFAULT_DENOISE,
    DEFAULT_SEED,
    DEFAULT_SEED_MODE,
    DEFAULT_VIDEO_DIMENSIONS,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_FRAMES,
    DEFAULT_WAN_BOUNDARY,
    DEFAULT_WAN_SHIFT,
    DEFAULT_WAN_VACE_EMPTY_FRAME_LEVEL,
    DEFAULT_WAN_VACE_STRENGTH,
    DENOISE_MAX,
    DENOISE_MIN,
    DENOISE_STEP,
    MAX_SEED,
    SEED_MIN,
    SEED_MODES,
    STEPS_MAX,
    VIDEO_DIMENSIONS,
    VIDEO_FPS_MAX,
    VIDEO_FPS_MIN,
    VIDEO_FPS_STEP,
    VIDEO_FRAMES_MAX,
    VIDEO_FRAMES_MIN,
    VIDEO_FRAMES_STEP,
    WAN_BOUNDARY_MAX,
    WAN_BOUNDARY_MIN,
    WAN_BOUNDARY_STEP,
    WAN_SHIFT_MAX,
    WAN_SHIFT_MIN,
    WAN_SHIFT_STEP,
    WAN_VACE_EMPTY_FRAME_LEVEL_MAX,
    WAN_VACE_EMPTY_FRAME_LEVEL_MIN,
    WAN_VACE_EMPTY_FRAME_LEVEL_STEP,
    WAN_VACE_STRENGTH_MAX,
    WAN_VACE_STRENGTH_MIN,
    WAN_VACE_STRENGTH_STEP,
    STEPS_MIN,
    STEPS_STEP,
)

InputSpec: TypeAlias = tuple[Any, ...]
InputGroup: TypeAlias = dict[str, InputSpec]
InputTypes: TypeAlias = dict[str, InputGroup]


class Input:
    @staticmethod
    def int(default: int, minimum: int, maximum: int, step: int = 1) -> InputSpec:
        return ("INT", {"default": default, "min": minimum, "max": maximum, "step": step})

    @staticmethod
    def float(default: float, minimum: float, maximum: float, step: float) -> InputSpec:
        return ("FLOAT", {"default": default, "min": minimum, "max": maximum, "step": step})

    @staticmethod
    def bool(default: bool = False) -> InputSpec:
        return ("BOOLEAN", {"default": default})

    @staticmethod
    def string(default: str = "", multiline: bool = False, force_input: bool = False) -> InputSpec:
        options: dict[str, Any] = {"default": default}

        if multiline:
            options["multiline"] = True

        if force_input:
            options["forceInput"] = True

        return ("STRING", options)

    @staticmethod
    def force(value_type: str) -> InputSpec:
        return (value_type, {"forceInput": True})

    @staticmethod
    def choice(values: list[str] | tuple[str, ...], default: str | None = None) -> InputSpec:
        if default is None:
            return (values,)

        return (values, {"default": default})

    @staticmethod
    def seed() -> InputSpec:
        return Input.int(DEFAULT_SEED, SEED_MIN, MAX_SEED)

    @staticmethod
    def seed_mode() -> InputSpec:
        return Input.choice(SEED_MODES, DEFAULT_SEED_MODE)

    @staticmethod
    def steps(default: int) -> InputSpec:
        return Input.int(default, STEPS_MIN, STEPS_MAX, STEPS_STEP)

    @staticmethod
    def cfg(default: float = DEFAULT_CFG) -> InputSpec:
        return Input.float(default, CFG_MIN, CFG_MAX, CFG_STEP)

    @staticmethod
    def denoise(default: float = DEFAULT_DENOISE) -> InputSpec:
        return Input.float(default, DENOISE_MIN, DENOISE_MAX, DENOISE_STEP)

    @staticmethod
    def video_dimensions(default: str = DEFAULT_VIDEO_DIMENSIONS) -> InputSpec:
        return Input.choice(VIDEO_DIMENSIONS, default)

    @staticmethod
    def video_frames(default: int = DEFAULT_VIDEO_FRAMES) -> InputSpec:
        return Input.int(default, VIDEO_FRAMES_MIN, VIDEO_FRAMES_MAX, VIDEO_FRAMES_STEP)

    @staticmethod
    def video_fps(default: int = DEFAULT_VIDEO_FPS) -> InputSpec:
        return Input.int(default, VIDEO_FPS_MIN, VIDEO_FPS_MAX, VIDEO_FPS_STEP)

    @staticmethod
    def wan_shift(default: float = DEFAULT_WAN_SHIFT) -> InputSpec:
        return Input.float(default, WAN_SHIFT_MIN, WAN_SHIFT_MAX, WAN_SHIFT_STEP)

    @staticmethod
    def wan_boundary(default: float = DEFAULT_WAN_BOUNDARY) -> InputSpec:
        return Input.float(default, WAN_BOUNDARY_MIN, WAN_BOUNDARY_MAX, WAN_BOUNDARY_STEP)

    @staticmethod
    def wan_vace_empty_frame_level(default: float = DEFAULT_WAN_VACE_EMPTY_FRAME_LEVEL) -> InputSpec:
        return Input.float(
            default,
            WAN_VACE_EMPTY_FRAME_LEVEL_MIN,
            WAN_VACE_EMPTY_FRAME_LEVEL_MAX,
            WAN_VACE_EMPTY_FRAME_LEVEL_STEP,
        )

    @staticmethod
    def wan_vace_strength(default: float = DEFAULT_WAN_VACE_STRENGTH) -> InputSpec:
        return Input.float(default, WAN_VACE_STRENGTH_MIN, WAN_VACE_STRENGTH_MAX, WAN_VACE_STRENGTH_STEP)

    @staticmethod
    def required(**inputs: InputSpec) -> InputTypes:
        return {"required": dict(inputs)}

    @staticmethod
    def optional(**inputs: InputSpec) -> InputTypes:
        return {"optional": dict(inputs)}

    @staticmethod
    def hidden(**inputs: str) -> InputTypes:
        return {"hidden": dict(inputs)}

    @staticmethod
    def groups(
        *,
        required: InputGroup | None = None,
        optional: InputGroup | None = None,
        hidden: dict[str, str] | None = None,
    ) -> InputTypes:
        result: InputTypes = {}

        if required is not None:
            result["required"] = required

        if optional is not None:
            result["optional"] = optional

        if hidden is not None:
            result["hidden"] = hidden

        return result


def force_input(value_type: str) -> InputSpec:
    return Input.force(value_type)
