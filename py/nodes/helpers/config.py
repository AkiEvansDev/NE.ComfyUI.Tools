from __future__ import annotations

from typing import Any

import comfy.samplers

from ....constants import *
from ...core.base import VideoConfig
from ...core.models import model_names
from .inputs import Input
from .formatting import round_float, parse_dimensions


def sampler_choices(default: str = DEFAULT_SAMPLER) -> tuple[Any, dict[str, str]]:
    return Input.choice(comfy.samplers.KSampler.SAMPLERS, default)


def scheduler_choices(default: str = DEFAULT_SCHEDULER) -> tuple[Any, dict[str, str]]:
    return Input.choice(comfy.samplers.KSampler.SCHEDULERS, default)


def upscale_model_choices() -> tuple[Any, ...]:
    names = model_names(FOLDER_UPSCALE_MODELS)
    # Only when it is installed: a default missing from the list is a value the prompt then fails to validate.
    return Input.choice(names, DEFAULT_UPSCALE_MODEL if DEFAULT_UPSCALE_MODEL in names else None)


def seed_inputs(*, even_default: bool) -> dict[str, tuple[Any, ...]]:
    return {
        "seed_value": Input.seed(),
        "mode": Input.seed_mode(),
        "even": Input.bool(even_default),
    }


def sampler_inputs(
    *,
    sampler_default: str = DEFAULT_SAMPLER,
    scheduler_default: str = DEFAULT_SCHEDULER,
    steps_default: int = DEFAULT_STEPS,
    cfg_default: float = DEFAULT_CFG,
    denoise_default: float | None = None,
) -> dict[str, tuple[Any, ...]]:
    inputs = {
        "sampler": sampler_choices(sampler_default),
        "scheduler": scheduler_choices(scheduler_default),
        "steps": Input.steps(steps_default),
        "cfg": Input.cfg(cfg_default),
    }

    if denoise_default is not None:
        inputs["denoise"] = Input.denoise(denoise_default)

    return inputs


def make_video_config(dimensions: str, frames: int, fps: int, shift: float) -> VideoConfig:
    width, height = parse_dimensions(dimensions)
    return VideoConfig(
        dimensions=dimensions,
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        shift=round_float(shift),
    )
