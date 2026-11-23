from __future__ import annotations

from nodes import EmptyLatentImage

from ...constants import *
from ..core.base import HiresConfig, ImageUpscaleConfig, ModelConfig, PipelineInfo, VideoConfig, VideoModelConfig, extract_filename
from .helpers import (
    Input,
    append_info,
    decoded_image,
    generation_info,
    hires_info,
    image_upscale_info,
    make_sampler_config,
    make_video_config,
    make_video_upscale_config,
    parse_dimensions,
    pipeline_info,
    resolve_seed,
    round_float,
    sample,
    sample_hires,
    sampler_inputs,
    seed_change_token,
    seed_inputs,
    upscale_image_with_config,
    upscale_model_choices,
    upscale_video_images_with_config,
    video_upscale_info,
    video_upscale_param_inputs,
)
from .seed import Seed
from .video import wan_image_to_video, wan_vace_to_video
from ..core.wan import decode_video_latent


def _latent_size(latent) -> tuple[int, int]:
    samples = latent["samples"]
    return samples.shape[3] * 8, samples.shape[2] * 8


def _upscaled_name(name: str | None, resize: float, model: str | None = None) -> str:
    model_name = extract_filename(model) if model else None
    suffix = f"upscale_{model_name}" if model_name else f"x{resize}"
    if not name:
        return suffix
    return f"{name}_{suffix}"


class GenerateStep:
    def __init__(self):
        self.seed = Seed()

    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "active": Input.bool(True),
                "model_config": ("MODEL_CONFIG",),
                "dimensions": Input.choice(DIMENSIONS, DEFAULT_DIMENSIONS),
                **seed_inputs(even_default=True),
                # denoise comes last out of the sampler block, and the block itself comes last on the
                # node: saved workflows restore widget values by position.
                **sampler_inputs(denoise_default=DEFAULT_DENOISE),
            },
            optional={
                "latent": ("LATENT",),
                "prompt": Input.force("STRING"),
            },
            hidden={"unique_id": "UNIQUE_ID"},
        )

    RETURN_TYPES = ("LATENT", "SAMPLER_CONFIG", "PIPELINE_INFO")
    RETURN_NAMES = ("latent", "config", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY_PIPELINE

    def run(
        self,
        active,
        model_config: ModelConfig,
        dimensions,
        seed_value,
        mode,
        even,
        sampler,
        scheduler,
        steps,
        cfg,
        denoise,
        unique_id,
        latent=None,
        prompt=None,
    ):
        seed_value = resolve_seed(self.seed, seed_value, mode, even, unique_id)
        config = make_sampler_config(
            seed=seed_value,
            sampler=sampler,
            scheduler=scheduler,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
        )

        if latent is None:
            width, height = parse_dimensions(dimensions)
            latent, = EmptyLatentImage().generate(width, height, 1)
            source = None
        else:
            width, height = _latent_size(latent)
            source = "latent + mask" if "noise_mask" in latent else "latent"

        if active:
            latent, = sample(model_config.model, model_config.positive, model_config.negative, latent, config)

        info = generation_info(
            active=active,
            width=width,
            height=height,
            config=config,
            source=source,
            model_name=model_config.model_name or None,
            loras_name=model_config.loras_name or None,
        )
        return (latent, config, PipelineInfo(text=append_info(prompt, info), name=f"{config.seed}"))

    @classmethod
    def IS_CHANGED(cls, mode, **kwargs):
        return seed_change_token(mode)


class HiresStep:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "active": Input.bool(True),
                "model_config": ("MODEL_CONFIG",),
                "latent": ("LATENT",),
                "base_config": ("SAMPLER_CONFIG",),
                "scale": Input.float(DEFAULT_HIRES_SCALE, HIRES_SCALE_MIN, HIRES_SCALE_MAX, HIRES_SCALE_STEP),
                "upscale_mode": Input.choice(UPSCALE_LATENT_MODES, DEFAULT_UPSCALE_LATENT_MODE),
                **sampler_inputs(
                    sampler_default=DEFAULT_HIRES_SAMPLER,
                    scheduler_default=DEFAULT_HIRES_SCHEDULER,
                    steps_default=DEFAULT_HIRES_STEPS,
                    cfg_default=DEFAULT_HIRES_CFG,
                    denoise_default=DEFAULT_HIRES_DENOISE,
                ),
            },
            optional={
                "info": ("PIPELINE_INFO",),
            },
        )

    RETURN_TYPES = ("LATENT", "PIPELINE_INFO")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY_PIPELINE

    def run(
        self,
        active,
        model_config: ModelConfig,
        latent,
        base_config,
        scale,
        upscale_mode,
        sampler,
        scheduler,
        steps,
        cfg,
        denoise,
        info=None,
    ):
        config = HiresConfig(
            seed=base_config.seed,
            scale=round_float(scale),
            upscale_mode=upscale_mode,
            sampler=sampler,
            scheduler=scheduler,
            steps=steps,
            denoise=round_float(denoise),
            cfg=round_float(cfg),
        )

        if active:
            latent, = sample_hires(model_config.model, model_config.positive, model_config.negative, latent, config)

        return (latent, pipeline_info(info, hires_info(active=active, config=config, loras_name=model_config.loras_name or None)))


class UpscaleStep:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "active": Input.bool(True),
                "model": upscale_model_choices(),
                "upscale_method": Input.choice(IMAGE_RESIZE_METHODS, DEFAULT_IMAGE_RESIZE_METHOD),
                "resize": Input.float(
                    DEFAULT_IMAGE_UPSCALE_RESIZE,
                    IMAGE_UPSCALE_RESIZE_MIN,
                    IMAGE_UPSCALE_RESIZE_MAX,
                    IMAGE_UPSCALE_RESIZE_STEP,
                ),
            },
            # Either end of the pipeline: an image a detailer has already worked on, or a latent
            # straight from a sampler. Both are optional so neither order needs a second node.
            optional={
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "info": ("PIPELINE_INFO",),
            },
        )

    RETURN_TYPES = ("IMAGE", "PIPELINE_INFO")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY_PIPELINE

    def run(self, active, model, upscale_method, resize, image=None, latent=None, vae=None, info=None):
        config = ImageUpscaleConfig(model_name=model, upscale_method=upscale_method, resize=resize)
        image = decoded_image(image, latent, vae)

        if active:
            image, = upscale_image_with_config(image, config)

        return (image, pipeline_info(info, image_upscale_info(active=active, config=config)))


class WanImageToVideoStep:
    def __init__(self):
        self.seed = Seed()

    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                **seed_inputs(even_default=True),
                "video_model_config": ("VIDEO_MODEL_CONFIG",),
                "image": ("IMAGE",),
                "dimensions": Input.choice(VIDEO_DIMENSIONS, DEFAULT_VIDEO_DIMENSIONS),
                "frames": Input.video_frames(),
                "fps": Input.video_fps(),
                "shift": Input.wan_shift(DEFAULT_WAN_SHIFT),
                **_video_sampler_inputs(),
            },
            optional={
                "end_image": ("IMAGE",),
            },
            hidden={"unique_id": "UNIQUE_ID"},
        )

    RETURN_TYPES = ("LATENT", "VIDEO_CONFIG", "PIPELINE_INFO")
    RETURN_NAMES = ("latent", "video_config", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY_PIPELINE

    def run(
        self,
        seed_value,
        mode,
        even,
        video_model_config: VideoModelConfig,
        image,
        dimensions,
        frames,
        fps,
        shift,
        sampler,
        scheduler,
        steps,
        cfg,
        denoise,
        unique_id,
        end_image=None,
    ):
        seed_value = resolve_seed(self.seed, seed_value, mode, even, unique_id)
        video_config = make_video_config(dimensions, frames, fps, shift)
        sampler_config = make_sampler_config(seed=seed_value, sampler=sampler, scheduler=scheduler, steps=steps, cfg=cfg, denoise=denoise)

        latent, info = wan_image_to_video(
            model_config=video_model_config,
            image=image,
            video_config=video_config,
            sampler_config=sampler_config,
            end_image=end_image,
        )
        return (latent, video_config, info)

    @classmethod
    def IS_CHANGED(cls, mode, **kwargs):
        return seed_change_token(mode)


# The two Wan steps share a sampler block with its own defaults, and it has to stay identical on both:
# widget values are restored by position, so a divergence silently shifts a saved workflow.
def _video_sampler_inputs() -> dict:
    return sampler_inputs(
        sampler_default=DEFAULT_VIDEO_SAMPLER,
        scheduler_default=DEFAULT_VIDEO_SCHEDULER,
        steps_default=DEFAULT_VIDEO_STEPS,
        cfg_default=DEFAULT_VIDEO_CFG,
        denoise_default=DEFAULT_DENOISE,
    )


class WanVaceToVideoStep:
    def __init__(self):
        self.seed = Seed()

    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                **seed_inputs(even_default=True),
                "video_model_config": ("VIDEO_MODEL_CONFIG",),
                "dimensions": Input.choice(VIDEO_DIMENSIONS, DEFAULT_VIDEO_DIMENSIONS),
                "frames": Input.video_frames(),
                "fps": Input.video_fps(),
                "shift": Input.wan_shift(DEFAULT_WAN_SHIFT),
                "empty_frame_level": Input.wan_vace_empty_frame_level(DEFAULT_WAN_VACE_EMPTY_FRAME_LEVEL),
                "strength": Input.wan_vace_strength(DEFAULT_WAN_VACE_STRENGTH),
                **_video_sampler_inputs(),
            },
            optional={
                "image": ("IMAGE",),
                "end_image": ("IMAGE",),
            },
            hidden={"unique_id": "UNIQUE_ID"},
        )

    RETURN_TYPES = ("LATENT", "VIDEO_CONFIG", "PIPELINE_INFO")
    RETURN_NAMES = ("latent", "video_config", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY_PIPELINE

    def run(
        self,
        seed_value,
        mode,
        even,
        video_model_config: VideoModelConfig,
        dimensions,
        frames,
        fps,
        shift,
        empty_frame_level,
        strength,
        sampler,
        scheduler,
        steps,
        cfg,
        denoise,
        unique_id,
        image=None,
        end_image=None,
    ):
        seed_value = resolve_seed(self.seed, seed_value, mode, even, unique_id)
        video_config = make_video_config(dimensions, frames, fps, shift)
        sampler_config = make_sampler_config(seed=seed_value, sampler=sampler, scheduler=scheduler, steps=steps, cfg=cfg, denoise=denoise)

        latent, info = wan_vace_to_video(
            model_config=video_model_config,
            video_config=video_config,
            sampler_config=sampler_config,
            empty_frame_level=empty_frame_level,
            strength=strength,
            image=image,
            end_image=end_image,
        )
        return (latent, video_config, info)

    @classmethod
    def IS_CHANGED(cls, mode, **kwargs):
        return seed_change_token(mode)


class VideoUpscaleStep:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "active": Input.bool(True),
                "images": ("IMAGE",),
                "video_config": ("VIDEO_CONFIG",),
                **video_upscale_param_inputs(),
            },
            optional={
                "info": ("PIPELINE_INFO",),
            },
        )

    RETURN_TYPES = ("IMAGE", "VIDEO_CONFIG", "PIPELINE_INFO")
    RETURN_NAMES = ("images", "video_config", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY_PIPELINE

    # params carries the widgets from video_upscale_param_inputs(), which are exactly the
    # keyword arguments make_video_upscale_config and video_upscale_info expect.
    def run(self, active, images, video_config: VideoConfig, info=None, **params):
        images, info = _run_video_upscale(active, images, info, params)
        return (images, video_config, info)


def _run_video_upscale(active: bool, images, info, params: dict):
    config = make_video_upscale_config(**params)
    name = None

    if active:
        name = _upscaled_name(info.name if info else None, config.resize, config.model_name)
        images = upscale_video_images_with_config(images, config)

    return images, pipeline_info(info, video_upscale_info(active=active, config=config), name)


class VideoLatentUpscaleStep:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "active": Input.bool(True),
                "latent": ("LATENT",),
                # The vae comes in the bundle, as it does on VideoDecode: one link for the same object.
                "video_model_config": ("VIDEO_MODEL_CONFIG",),
                "video_config": ("VIDEO_CONFIG",),
                **video_upscale_param_inputs(),
            },
            optional={
                "info": ("PIPELINE_INFO",),
            },
        )

    RETURN_TYPES = ("IMAGE", "VIDEO_CONFIG", "PIPELINE_INFO")
    RETURN_NAMES = ("images", "video_config", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY_PIPELINE

    def run(self, active, latent, video_model_config: VideoModelConfig, video_config: VideoConfig, info=None, **params):
        images, = decode_video_latent(video_model_config.vae, latent)
        images, info = _run_video_upscale(active, images, info, params)
        return (images, video_config, info)
