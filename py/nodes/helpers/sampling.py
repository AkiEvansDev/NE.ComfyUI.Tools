from __future__ import annotations

from typing import Any

import comfy.samplers
from nodes import KSampler, KSamplerAdvanced, LatentUpscaleBy, VAEDecode, VAEEncode

from ....constants import DEFAULT_DENOISE, UPSCALE_LATENT_CUSTOM_MODE
from ...core.base import HiresConfig, SamplerConfig
from .formatting import round_float
from ..upscale import UpscaleLatentBy

Conditioning = Any
Latent = Any
Model = Any
Vae = Any
Image = Any
Sigmas = Any


def make_sampler_config(
    *,
    seed: int,
    sampler: str,
    scheduler: str,
    steps: int,
    cfg: float,
    denoise: float = DEFAULT_DENOISE,
) -> SamplerConfig:
    return SamplerConfig(
        seed=seed,
        sampler=sampler,
        scheduler=scheduler,
        steps=steps,
        cfg=round_float(cfg),
        denoise=round_float(denoise),
    )


def sample(model: Model, positive: Conditioning, negative: Conditioning, latent: Latent, config: SamplerConfig) -> tuple[Latent]:
    return KSampler().sample(
        model,
        config.seed,
        config.steps,
        config.cfg,
        config.sampler,
        config.scheduler,
        positive,
        negative,
        latent,
        config.denoise,
    )


def sample_split(model_high: Model, model_low: Model, positive: Conditioning, negative: Conditioning, latent: Latent,
                 config: SamplerConfig, switch_step: int) -> tuple[Latent]:
    """Wan 2.2's two experts over one schedule: the high-noise model runs its front, the low-noise one the rest."""
    latent, = KSamplerAdvanced().sample(
        model_high, "enable", config.seed, config.steps, config.cfg, config.sampler, config.scheduler,
        positive, negative, latent, 0, switch_step, "enable", denoise=config.denoise,
    )
    # add_noise disabled and the leftover noise carried over: the second stage continues the first one's
    # schedule rather than starting its own.
    return KSamplerAdvanced().sample(
        model_low, "disable", config.seed, config.steps, config.cfg, config.sampler, config.scheduler,
        positive, negative, latent, switch_step, config.steps, "disable", denoise=config.denoise,
    )


def split_step(model: Model, config: SamplerConfig, boundary: float) -> int:
    """Where the Wan 2.2 high-noise expert hands over, on the schedule this run is about to use."""
    return _split_step(_schedule_sigmas(model, config), config.steps, boundary)


def _schedule_sigmas(model: Model, config: SamplerConfig) -> Sigmas:
    """The sigma schedule the sampler is about to run, denoise trimming and sampler quirks included."""
    # Built through comfy's own KSampler rather than calculate_sigmas directly: the denoise trimming and the
    # penultimate-sigma discard for dpm_2/uni_pc live in there, and a second copy of them would drift apart.
    sampler = comfy.samplers.KSampler(
        model, config.steps, "cpu", sampler=config.sampler, scheduler=config.scheduler, denoise=config.denoise,
    )
    return sampler.sigmas


def _split_step(sigmas: Sigmas, steps: int, boundary: float) -> int:
    """Where the high-noise expert hands over — the first step that starts below the boundary sigma."""
    # Wan states the handover as a fraction of the timestep range, and a flow model's sigma is exactly that
    # fraction (ModelSamplingDiscreteFlow.timestep is sigma * multiplier), so the two compare directly.
    # Splitting on the step index instead quantized the setting to 1/steps, which is three usable values at
    # the four steps these distills are made for, and moved with `shift` without saying so.
    above = sum(1 for sigma in sigmas[:-1] if float(sigma) >= boundary)
    # A stage of zero steps means one of the two models was loaded and never used, which reads as a hang
    # rather than as a setting: every boundary still spends one step on each expert.
    return max(1, min(steps - 1, above))


def decoded_image(image: Image | None, latent: Latent | None, vae: Vae | None) -> Image:
    """An image either handed over directly, or decoded from a latent — the two ways a step gets pixels."""
    if image is not None:
        return image

    if latent is None or vae is None:
        raise ValueError("Connect either an image, or a latent together with a vae to decode it.")

    decoded, = VAEDecode().decode(vae, latent)
    return decoded


def encode_image_to_latent(vae: Vae, image: Image) -> Latent:
    latent, = VAEEncode().encode(vae, image)
    return latent


def sample_hires(model: Model, positive: Conditioning, negative: Conditioning, latent: Latent, config: HiresConfig) -> tuple[Latent]:
    latent, = _upscale_latent(latent, config)
    return sample(model, positive, negative, latent, config)


def _upscale_latent(latent: Latent, config: HiresConfig) -> tuple[Latent]:
    if config.upscale_mode == UPSCALE_LATENT_CUSTOM_MODE:
        return UpscaleLatentBy().upscale(latent, config.scale)

    return LatentUpscaleBy().upscale(latent, config.upscale_mode, config.scale)
