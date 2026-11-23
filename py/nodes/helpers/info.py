"""Info blocks for the pipeline steps.

Values left as None are dropped by format_info, which is how a step that ran without a source or
without lora names still produces a tidy block.
"""

from __future__ import annotations

from typing import Any

from ...core.base import HiresConfig, ImageUpscaleConfig, SamplerConfig, VideoConfig, VideoModelConfig, extract_filename
from .formatting import format_info


def generation_info(
    *,
    active: bool,
    width: int,
    height: int,
    config: SamplerConfig,
    source: str | None = None,
    model_name: str | None = None,
    loras_name: str | None = None,
) -> str:
    return format_info(
        "Generate",
        {
            "Active": active,
            "Source": source,
            "Dimensions": f"{width}x{height}",
            "Seed": config.seed,
            "Sampler": config.sampler,
            "Scheduler": config.scheduler,
            "Steps": config.steps,
            "Cfg": config.cfg,
            "Denoise": config.denoise,
            "Model": model_name,
            "Loras": f"\n{loras_name}" if loras_name else None,
        },
    )


def wan_generation_info(*, mode: str, model_config: VideoModelConfig, video_config: VideoConfig,
                        sampler_config: SamplerConfig, switch_step: int | None = None,
                        clip_vision: str | None = None) -> str:
    return _wan_info(
        "Wan Video",
        mode,
        model_config,
        video_config,
        sampler_config,
        switch_step,
        {"ClipVision": clip_vision},
    )


def wan_vace_generation_info(*, mode: str, model_config: VideoModelConfig, video_config: VideoConfig,
                             sampler_config: SamplerConfig, empty_frame_level: float, strength: float,
                             switch_step: int | None = None) -> str:
    return _wan_info(
        "Wan VACE Video",
        mode,
        model_config,
        video_config,
        sampler_config,
        switch_step,
        {"EmptyFrameLevel": empty_frame_level, "Strength": strength},
    )


def _wan_info(title: str, mode: str, model_config: VideoModelConfig, video_config: VideoConfig,
              sampler_config: SamplerConfig, switch_step: int | None, extra: dict[str, Any]) -> str:
    """The prompts, then one block — the video counterpart of generation_info, model and lora names included."""
    return "\n".join(
        [
            "[Positive]",
            model_config.positive_text,
            "",
            "[Negative]",
            model_config.negative_text,
            "",
            format_info(
                title,
                {
                    "Mode": mode,
                    "Dimensions": f"{video_config.width}x{video_config.height}",
                    "Frames": video_config.frames,
                    "FPS": video_config.fps,
                    "Duration": f"{video_config.duration:.2f}s",
                    "Shift": video_config.shift,
                    "Boundary": _boundary_info(model_config.boundary, switch_step, sampler_config.steps),
                    **extra,
                    "Seed": sampler_config.seed,
                    "Sampler": sampler_config.sampler,
                    "Scheduler": sampler_config.scheduler,
                    "Steps": sampler_config.steps,
                    "Cfg": sampler_config.cfg,
                    "Denoise": sampler_config.denoise,
                    "Model": model_config.model_name or None,
                    "Loras": f"\n{model_config.loras_name}" if model_config.loras_name else None,
                },
            ),
        ]
    )


def _boundary_info(boundary: float, switch_step: int | None, steps: int) -> str | None:
    """The boundary sigma together with the split it produced."""
    # Only a two-expert run splits anything; on a single model the row would read as a setting that did
    # something. A run too short to split reports nothing for the same reason.
    if switch_step is None:
        return None

    # Which split a boundary lands on depends on shift, steps and scheduler, so the number on its own is
    # not tunable — reading it back would mean rebuilding the schedule by hand.
    return f"{boundary} (high {switch_step}/{steps})"


def hires_info(*, active: bool, config: HiresConfig, loras_name: str | None = None) -> str:
    return format_info(
        "Hires",
        {
            "Active": active,
            "Scale": config.scale,
            "Mode": config.upscale_mode,
            "Sampler": config.sampler,
            "Scheduler": config.scheduler,
            "Steps": config.steps,
            "Denoise": config.denoise,
            "Cfg": config.cfg,
            "Loras": f"\n{loras_name}" if loras_name else None,
        },
    )


def image_upscale_info(*, active: bool, config: ImageUpscaleConfig) -> str:
    return format_info(
        "Image Upscale + Resize",
        {
            "Active": active,
            "Model": extract_filename(config.model_name),
            "Method": config.upscale_method,
            "Resize": config.resize,
        },
    )
