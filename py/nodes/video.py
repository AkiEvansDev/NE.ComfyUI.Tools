from __future__ import annotations

from typing import Any

from ...constants import *
from ..core.base import PipelineInfo, SamplerConfig, VideoConfig, VideoModelConfig
from ..core.wan import (
    apply_model_sampling_sd3,
    clip_vision_encode,
    decode_video_latent,
    model_takes_clip_vision,
)
from ..core.wan.wan_nodes_utility import WanVideoVACEStartToEndFrame
from comfy_extras.nodes_wan import WanFirstLastFrameToVideo as ComfyWanFirstLastFrameToVideo
from comfy_extras.nodes_wan import WanImageToVideo as ComfyWanImageToVideo
from comfy_extras.nodes_wan import WanVaceToVideo as ComfyWanVaceToVideo
from .helpers import (
    Input,
    round_float,
    sample,
    sample_split,
    split_step,
    wan_generation_info,
    wan_vace_generation_info,
)


def wan_image_to_video(
    *,
    model_config: VideoModelConfig,
    image,
    video_config: VideoConfig,
    sampler_config: SamplerConfig,
    end_image=None,
) -> tuple[Any, PipelineInfo]:
    positive, negative = model_config.positive, model_config.negative
    clip_vision = model_config.clip_vision

    # Wan 2.1 I2V projects the CLIP Vision embedding through img_emb; Wan 2.2 14B has no img_emb, and an
    # embedding handed to it makes the model reserve image tokens the context never got. So the encoder may
    # stay wired to both and the model decides — leaving one graph that works for either generation.
    use_clip_vision = clip_vision is not None and model_takes_clip_vision(model_config.model)
    start_embed = clip_vision_encode(clip_vision, image) if use_clip_vision else None

    if end_image is None:
        positive, negative, latent = _node_output_values(
            ComfyWanImageToVideo.execute(
                positive,
                negative,
                model_config.vae,
                video_config.width,
                video_config.height,
                video_config.frames,
                1,
                start_image=image,
                clip_vision_output=start_embed,
            ),
            3,
        )
    else:
        positive, negative, latent = _node_output_values(
            ComfyWanFirstLastFrameToVideo.execute(
                positive,
                negative,
                model_config.vae,
                video_config.width,
                video_config.height,
                video_config.frames,
                1,
                start_image=image,
                end_image=end_image,
                clip_vision_start_image=start_embed,
                clip_vision_end_image=clip_vision_encode(clip_vision, end_image) if use_clip_vision else None,
            ),
            3,
        )
    latent, switch_step = sample_experts(model_config, positive, negative, latent, sampler_config, video_config.shift)

    mode = "first-last-frame" if end_image is not None else "image-to-video"
    block = wan_generation_info(
        mode=mode,
        model_config=model_config,
        video_config=video_config,
        sampler_config=sampler_config,
        switch_step=switch_step,
        clip_vision=_clip_vision_state(clip_vision, use_clip_vision),
    )
    name = f"wan_{video_config.width}x{video_config.height}_{video_config.frames}f_{video_config.fps}fps"
    return latent, PipelineInfo(text=block, name=name)


def sample_experts(model_config: VideoModelConfig, positive, negative, latent, sampler_config: SamplerConfig,
                   shift: float) -> tuple[Any, int | None]:
    """One model or the Wan 2.2 pair, both shifted the same way — the only difference a second expert makes.

    The handover step comes back with the latent: it is decided here, from the shifted schedule, and the
    info block would otherwise have to rebuild that schedule to say what the boundary actually did.
    """
    sampled_model = apply_model_sampling_sd3(model_config.model, shift)

    # A single step cannot be split without handing one of the two experts nothing to do.
    if model_config.model_low is None or sampler_config.steps < 2:
        latent, = sample(sampled_model, positive, negative, latent, sampler_config)
        return latent, None

    sampled_low = apply_model_sampling_sd3(model_config.model_low, shift)
    switch_step = split_step(sampled_model, sampler_config, model_config.boundary)
    latent, = sample_split(sampled_model, sampled_low, positive, negative, latent, sampler_config, switch_step)
    return latent, switch_step


def _clip_vision_state(clip_vision, used: bool) -> str | None:
    """What happened to a connected CLIP Vision encoder — silently dropping one would be a mystery later."""
    if clip_vision is None:
        return None

    return "used" if used else "ignored, the model has no img_emb"


def _node_output_values(result, count: int):
    values = getattr(result, "result", result)

    if callable(values):
        values = values()

    if isinstance(values, tuple):
        return values

    if isinstance(values, list):
        return tuple(values)

    try:
        return tuple(values)
    except TypeError:
        return (values,) * count


def wan_vace_to_video(
    *,
    model_config: VideoModelConfig,
    video_config: VideoConfig,
    sampler_config: SamplerConfig,
    empty_frame_level: float,
    strength: float,
    image=None,
    end_image=None,
) -> tuple[Any, PipelineInfo]:
    positive, negative = model_config.positive, model_config.negative
    effective_strength = 0.0 if image is None and end_image is None else round_float(strength)

    if image is None and end_image is None:
        control_video = None
        mask = None
    else:
        control_video, mask = WanVideoVACEStartToEndFrame().process(
            video_config.frames,
            round_float(empty_frame_level),
            start_image=image,
            end_image=end_image,
        )

    positive, negative, latent, _trim_latent = _node_output_values(
        ComfyWanVaceToVideo.execute(
            positive,
            negative,
            model_config.vae,
            video_config.width,
            video_config.height,
            video_config.frames,
            1,
            effective_strength,
            control_video=control_video,
            control_masks=mask,
        ),
        4,
    )
    latent, switch_step = sample_experts(model_config, positive, negative, latent, sampler_config, video_config.shift)

    mode = _vace_mode(image, end_image)
    block = wan_vace_generation_info(
        mode=mode,
        model_config=model_config,
        video_config=video_config,
        sampler_config=sampler_config,
        switch_step=switch_step,
        empty_frame_level=round_float(empty_frame_level),
        strength=effective_strength,
    )
    name = f"wan_vace_{mode}_{video_config.width}x{video_config.height}_{video_config.frames}f_{video_config.fps}fps"
    return latent, PipelineInfo(text=block, name=name)


def _vace_mode(image, end_image) -> str:
    if image is not None and end_image is not None:
        return "start-end-frame"
    if image is not None:
        return "start-frame"
    if end_image is not None:
        return "end-frame"
    return "text-to-video"


class VideoDecode:
    @classmethod
    def INPUT_TYPES(cls):
        # The vae arrives inside the bundle: a second link for it would be the same object wired twice.
        return Input.required(
            latent=("LATENT",),
            video_model_config=("VIDEO_MODEL_CONFIG",),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_VIDEO

    def execute(self, latent, video_model_config: VideoModelConfig):
        return decode_video_latent(video_model_config.vae, latent)
