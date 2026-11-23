from __future__ import annotations

from dataclasses import replace

import torch
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
from nodes import CLIPTextEncode, ImageScale, SetLatentNoiseMask, VAEDecode

from ...constants import *
from ..core.base import ModelConfig, SamplerConfig, is_not_blank
from ..core.image import ensure_batch, ensure_mask_batch, scale_mask
from ..core.regions import Box, Region, crop_box, mask_regions
from .helpers import (
    Input,
    encode_image_to_latent,
    format_info,
    make_sampler_config,
    pipeline_info,
    round_float,
    sample,
    sampler_inputs,
)
from .image_filters import blur_mask_tensor
from .prompt import PromptText, RegionSections, build_region_prompt_text, parse_region_sections


class DetailerStep:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                # The region prompt describes the part, not the picture: only the two style parts of the
                # bundle cross over. See build_region_prompt_text for why. A `1: tags` line addresses one
                # region by the number the detect preview draws on it.
                "region_positive": Input.string(DEFAULT_DETAILER_REGION_POSITIVE, multiline=True),
                "region_negative": Input.string(DEFAULT_DETAILER_REGION_NEGATIVE, multiline=True),
                "active": Input.bool(True),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "model_config": ("MODEL_CONFIG",),
                "vae": ("VAE",),
                "base_config": ("SAMPLER_CONFIG",),
                "region_size": Input.int(
                    DEFAULT_DETAILER_REGION_SIZE,
                    DETAILER_REGION_SIZE_MIN,
                    DETAILER_REGION_SIZE_MAX,
                    DETAILER_REGION_SIZE_STEP,
                ),
                "max_size": Input.int(DEFAULT_DETAILER_MAX_SIZE, DETAILER_MAX_SIZE_MIN, DETAILER_MAX_SIZE_MAX, DETAILER_MAX_SIZE_STEP),
                "crop_factor": Input.float(
                    DEFAULT_DETAILER_CROP_FACTOR,
                    DETAILER_CROP_FACTOR_MIN,
                    DETAILER_CROP_FACTOR_MAX,
                    DETAILER_CROP_FACTOR_STEP,
                ),
                # Two feathers, not one: the paste wants a wide band to hide the edge of a box-shaped mask,
                # while the sampler wants a narrow one so the region itself is left free.
                "feather": Input.int(DEFAULT_DETAILER_FEATHER, DETAILER_FEATHER_MIN, DETAILER_FEATHER_MAX, DETAILER_FEATHER_STEP),
                "noise_feather": Input.int(
                    DEFAULT_DETAILER_NOISE_FEATHER,
                    DETAILER_FEATHER_MIN,
                    DETAILER_FEATHER_MAX,
                    DETAILER_FEATHER_STEP,
                ),
                "differential": Input.bool(DEFAULT_DETAILER_DIFFERENTIAL),
                **sampler_inputs(
                    sampler_default=DEFAULT_DETAILER_SAMPLER,
                    scheduler_default=DEFAULT_DETAILER_SCHEDULER,
                    steps_default=DEFAULT_DETAILER_STEPS,
                    cfg_default=DEFAULT_DETAILER_CFG,
                    denoise_default=DEFAULT_DETAILER_DENOISE,
                ),
                # Below the sampler block rather than beside base_config: they answer "which noise", which
                # is a question about the sampler run rather than about the region it is given.
                "seed_mode": Input.choice(DETAILER_SEED_MODES, DEFAULT_DETAILER_SEED_MODE),
                "seed_value": Input.seed(),
                "vary_per_region": Input.bool(DEFAULT_DETAILER_VARY_SEED),
                # Last, because saved workflows restore widget values by position.
                "cycles": Input.int(DEFAULT_DETAILER_CYCLES, DETAILER_CYCLES_MIN, DETAILER_CYCLES_MAX),
            },
            optional={
                "info": ("PIPELINE_INFO",),
            },
        )

    RETURN_TYPES = ("IMAGE", "MASK", "PIPELINE_INFO")
    RETURN_NAMES = ("image", "mask", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY_PIPELINE

    def run(
        self,
        region_positive,
        region_negative,
        active,
        image,
        mask,
        model_config: ModelConfig,
        vae,
        base_config,
        region_size,
        max_size,
        crop_factor,
        feather,
        noise_feather,
        differential,
        sampler,
        scheduler,
        steps,
        cfg,
        denoise,
        seed_mode,
        seed_value,
        vary_per_region,
        cycles=DEFAULT_DETAILER_CYCLES,
        info=None,
    ):
        config = make_sampler_config(
            seed=_base_seed(base_config.seed, seed_mode, seed_value),
            sampler=sampler,
            scheduler=scheduler,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
        )
        prompts = _RegionPrompts(model_config, region_positive, region_negative)
        # Without a band there is nothing for it to reinterpret: a hard mask is 0 or 1 either way.
        differential = differential and noise_feather > 0
        model = _detail_model(model_config.model, differential) if active else model_config.model

        image = ensure_batch(image)
        mask = ensure_mask_batch(mask)

        # A mask from a differently sized source would slice the frame at the wrong offsets.
        if mask.shape[1:] != image.shape[1:3]:
            mask = scale_mask(mask, image.shape[2], image.shape[1], DEFAULT_IMAGE_RESIZE_METHOD)

        result = image.clone()
        handled = torch.zeros(image.shape[:3], dtype=torch.float32)
        regions = 0
        skipped: set[int] = set()

        for index in range(image.shape[0]):
            # One mask for the whole batch is the common case; a per-frame mask is used when given.
            frame_mask = mask[min(index, mask.shape[0] - 1)]

            for index_in_frame, region in enumerate(mask_regions(frame_mask)):
                regions += 1
                # Numbered from one, the way the detect preview labels them.
                number = index_in_frame + 1

                # Left out of `handled` as well: the mask output says what the step touched, and a region
                # taken out by `2: -` is the one thing the caller asked it not to.
                if prompts.skipped(number):
                    skipped.add(number)
                    continue

                crop = _crop_for(region.box, image.shape[2], image.shape[1], region_size, max_size, crop_factor)
                if crop is None:
                    continue

                handled[index] = _paint_box(handled[index], _touched_box(region.box, crop, feather, image.shape[2], image.shape[1]))

                if not active:
                    continue

                positive, negative = prompts.conditioning(number)
                result[index] = self._detail_region(
                    result[index],
                    frame_mask,
                    region,
                    crop,
                    model=model,
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    config=_region_config(config, index_in_frame, vary_per_region),
                    region_size=region_size,
                    max_size=max_size,
                    feather=feather,
                    noise_feather=noise_feather,
                    cycles=cycles,
                )

        block = _detailer_info(
            active=active,
            regions=regions,
            skipped=sorted(skipped),
            region_size=region_size,
            max_size=max_size,
            crop_factor=crop_factor,
            feather=feather,
            noise_feather=noise_feather,
            differential=differential,
            config=config,
            seed_mode=seed_mode,
            vary_per_region=vary_per_region,
            cycles=cycles,
            prompts=prompts,
            loras_name=model_config.loras_name,
        )
        return (result, handled, pipeline_info(info, block))

    def _detail_region(self, frame, frame_mask, region: Region, bounds: Box, *, model, positive, negative, vae, config,
                       region_size, max_size, feather, noise_feather, cycles):
        left, top, right, bottom = bounds
        crop = frame[top:bottom, left:right, :].unsqueeze(0)
        # This region's blob alone: the crop of one hand often holds the other, and a mask still carrying it
        # would have this pass repaint that hand too, at a scale chosen for this one, and the last pass win.
        crop_mask = _region_mask(frame_mask, region)[top:bottom, left:right].unsqueeze(0)
        crop_width, crop_height = right - left, bottom - top

        work_width, work_height = _working_size(region.box, bounds, region_size, max_size)
        work_image, = ImageScale().upscale(crop, DEFAULT_IMAGE_RESIZE_METHOD, work_width, work_height, "disabled")
        work_mask = scale_mask(crop_mask, work_width, work_height, DEFAULT_IMAGE_RESIZE_METHOD)

        latent = encode_image_to_latent(vae, work_image)
        # A band of its own, narrower than the paste: behind a hard-edged noise mask the region drifts away
        # from what surrounds it, but every pixel the band covers is one the sampler starts late on, so it
        # is kept off the region itself.
        latent, = SetLatentNoiseMask().set_mask(latent, _feathered(work_mask, round(noise_feather * work_width / crop_width)))
        # Each cycle continues from the last one's latent, noise mask and all, with noise of its own.
        for cycle in range(cycles):
            latent, = sample(model, positive, negative, latent, _cycle_config(config, cycle))

        detailed, = VAEDecode().decode(vae, latent)
        detailed, = ImageScale().upscale(detailed, DEFAULT_IMAGE_RESIZE_METHOD, crop_width, crop_height, "disabled")

        # Everything the sampler was allowed to touch goes back at full weight and the ramp runs outward through
        # the context, where it was held to the original. Pasting by the soft value instead is a cross-fade of
        # the old hand and the new one - the double exposure a soft or shaped mask would otherwise bring back.
        band = _paste_band(region.box, bounds, feather, frame.shape[1], frame.shape[0])
        blend = _paste_mask((crop_mask > 0).float(), band)[0].unsqueeze(-1)
        frame = frame.clone()
        frame[top:bottom, left:right, :] = crop[0] * (1.0 - blend) + detailed[0] * blend
        return frame


def _base_seed(config_seed: int, mode: str, seed_value: int) -> int:
    """The pipeline seed, one typed here, or the two added - so a detail pass can be re-rolled on its own."""
    if mode == DETAILER_SEED_MODE_FIXED:
        return seed_value

    if mode == DETAILER_SEED_MODE_OFFSET:
        return min(config_seed + seed_value, MAX_SEED)

    return config_seed


class _RegionPrompts:
    """A crop is conditioned on its own prompt, built from the shared parts the bundle came with.

    Each region can carry text of its own: a `1: tags` section in either widget applies to that region
    alone, and whatever stands before the first section applies to all of them. The encoding is cached by
    the text it produces rather than by the number - without sections every region asks for the same pair,
    and a batch asks for it once per frame.
    """
    def __init__(self, model_config: ModelConfig, region_positive: str, region_negative: str) -> None:
        self.clip = model_config.clip
        self.config = model_config.prompt
        self.positive = parse_region_sections(region_positive)
        self.negative = parse_region_sections(region_negative)
        # Region numbers start at one, so zero is the text a region without a section of its own gets.
        self.shared = self._text(0)
        self.encoded: dict[tuple[str, str], tuple] = {}

    def conditioning(self, number: int) -> tuple:
        text = self._text(number) if self._has_section(number) else self.shared
        key = (text.positive, text.negative)

        if key not in self.encoded:
            # Checked here rather than up front: a switched-off step encodes nothing and must not demand a clip.
            if self.clip is None:
                raise ValueError("The detailer needs a model config carrying a clip, otherwise it cannot encode a region prompt.")

            encoder = CLIPTextEncode()
            positive, = encoder.encode(self.clip, text.positive)
            negative, = encoder.encode(self.clip, text.negative)
            self.encoded[key] = (positive, negative)

        return self.encoded[key]

    def _text(self, number: int) -> PromptText:
        return build_region_prompt_text(self.config, self.positive.for_region(number), self.negative.for_region(number))

    def _has_section(self, number: int) -> bool:
        return number in self.positive.sections or number in self.negative.sections

    def skipped(self, number: int) -> bool:
        """Either widget carries the marker: whichever one is open when a region turns out to be unwanted."""
        return self.positive.skips_region(number) or self.negative.skips_region(number)


def _detail_model(model, differential: bool):
    """Differential diffusion, so a feathered noise mask reads as a start time rather than a blend weight.

    Comfy applies a soft denoise mask as `out = out * mask + original * (1 - mask)` on every step
    (`comfy/samplers.py`), which leaves the falloff band a permanent cross-fade of two pictures - the mush
    that kept this step under ~0.5 denoise. Differential diffusion turns the same value into the step the
    cell starts denoising at, so the band still anchors the region while the noise is high and is free
    again by the end, and it comes back sharp. Impact Pack does the same whenever its noise mask is
    feathered, and it is why the band may now reach into the region without softening it.
    """
    if not differential or "denoise_mask_function" in model.model_options:
        return model

    return DifferentialDiffusion.execute(model)[0]


def _region_config(config: SamplerConfig, region: int, vary_per_region: bool) -> SamplerConfig:
    """A seed of its own per region: two hands in one frame otherwise get the same noise and the same flaw.

    The index is the position within the frame, and mask_regions numbers regions the same way every run,
    so this still repeats exactly.
    """
    if not vary_per_region or region == 0:
        return config

    return replace(config, seed=min(config.seed + region, MAX_SEED))


def _crop_for(box: Box, width: int, height: int, region_size: int, max_size: int, crop_factor: float) -> Box | None:
    """The crop a region is sampled inside: `crop_factor` of context, less when that would cost resolution.

    The crop's long side is capped by `max_size` at the scale the region asks for, and with a fixed factor
    that cap binds first for nearly every region - 1024 / 3.0 left any region 341px at most, so a
    `region_size` of 512 was never reached and a 400px face was sampled 1:1 across most of the frame. The
    context gives way first, down to DETAILER_MIN_CROP_FACTOR, and only past that does the region lose pixels.
    """
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    # Not below 1:1, the floor _working_size keeps too: a region already past region_size is sampled as it
    # is, and context beyond what fits max_size at that scale only adds cost.
    scale = max(1.0, _region_scale(box, region_size))
    fitting = max_size / (scale * max(box_width, box_height))
    factor = max(min(crop_factor, fitting), min(crop_factor, DETAILER_MIN_CROP_FACTOR))
    return crop_box(box, width, height, factor)


def _region_scale(box: Box, region_size: int) -> float:
    """What the region's short side has to be multiplied by to reach region_size."""
    return region_size / max(1, min(box[2] - box[0], box[3] - box[1]))


def _cycle_config(config: SamplerConfig, cycle: int) -> SamplerConfig:
    """A seed of its own per cycle, stepped past every region's so a cycle never replays another region's noise."""
    if cycle == 0:
        return config

    return replace(config, seed=min(config.seed + cycle * DETAILER_MAX_REGIONS, MAX_SEED))


def _region_mask(frame_mask: torch.Tensor, region: Region) -> torch.Tensor:
    """The frame mask inside this region's box, with every other blob taken out of it.

    The box alone is not enough: blobs whose boxes overlap each hold part of the other. The soft values are
    kept rather than the blob itself, so a shaped or feathered mask still reaches the sampler as drawn.
    """
    left, top, right, bottom = region.box
    mask = torch.zeros_like(frame_mask)
    mask[top:bottom, left:right] = frame_mask[top:bottom, left:right]
    mask[(frame_mask >= DETAILER_MASK_THRESHOLD) & ~region.blob.to(frame_mask.device)] = 0.0
    return mask


def _working_size(box: Box, crop: Box, region_size: int, max_size: int) -> tuple[int, int]:
    """The resolution a crop is sampled at: enough for the region, capped for the crop, never below 1:1.

    The factor comes off the *region* rather than the crop, because the region is what has to end up with
    the latent cells to draw fingers in - taking it off the crop hands the least resolution to the smallest
    boxes, which are the ones a detail pass exists for. Its short side sets the factor, the crop's long
    side caps it.

    Scaling down is refused outright: the round trip back up afterwards is a blur, and asking the sampler
    to repair a blur it was handed is how a good hand and a broken background end up being one setting.
    """
    crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
    scale = max(1.0, min(_region_scale(box, region_size), max_size / max(crop_width, crop_height)))
    return _align_size(crop_width * scale), _align_size(crop_height * scale)


def _align_size(value: float) -> int:
    return max(8, int(round(value)) // 8 * 8)


def _feathered(mask: torch.Tensor, feather: int) -> torch.Tensor:
    """A (B, H, W) mask with a soft edge. The radius is in pixels of whatever resolution it is given at.

    The blur is symmetric, so it reaches `feather` pixels *inside* the region as well, and on the noise
    mask that is the anchor holding the region onto what surrounds it - under differential diffusion the
    inward ramp is a late start, not a blend, so it costs no sharpness. The paste does not use this edge
    as it stands: see `_paste_mask`.

    The solid interior is snapped to exactly 1.0. A blur leaves it at 1 - 1e-7, and differential diffusion
    frees a cell on the first step only when its mask is >= 1.0 exactly - so without the snap the first
    step of every region ran on nothing at all.
    """
    soft = blur_mask_tensor(mask, feather, max(0.1, feather / 2.0)).squeeze(-1)
    return torch.where(soft >= 1.0 - DETAILER_MASK_SOLID_EPSILON, torch.ones_like(soft), soft)


def _paste_band(box: Box, crop: Box, feather: int, width: int, height: int) -> int:
    """How far the paste ramp may reach past the box: `feather`, unless the crop ends sooner.

    The ramp runs outward over two radii, and one that is cut off by the crop edge leaves a step there
    instead of a fade. A crop edge that is the frame edge does not count - there is nothing beyond it.
    """
    left, top, right, bottom = box
    crop_left, crop_top, crop_right, crop_bottom = crop
    margins = [feather * 2]
    if crop_left > 0:
        margins.append(left - crop_left)
    if crop_top > 0:
        margins.append(top - crop_top)
    if crop_right < width:
        margins.append(crop_right - right)
    if crop_bottom < height:
        margins.append(crop_bottom - bottom)
    return max(0, min(margins) // 2)


def _paste_mask(mask: torch.Tensor, band: int) -> torch.Tensor:
    """The blend weight for pasting a region back: solid over the box, fading out over the context.

    Grown by the band first and blurred by it after, so the region itself is pasted whole and the fade lies
    over pixels the sampler was held to the original on. This is deliberately the opposite of the noise
    mask: a fade reaching into the region mixes the old hand back into the new one over that band, and
    that cross-fade is the soft edge a detailed region used to show.
    """
    if band <= 0:
        return mask

    grown = torch.nn.functional.max_pool2d(mask.unsqueeze(1), kernel_size=band * 2 + 1, stride=1, padding=band).squeeze(1)
    return _feathered(grown, band)


def _touched_box(box: Box, crop: Box, feather: int, width: int, height: int) -> Box:
    """What the paste actually changes: the box plus the two radii of ramp around it, inside the crop.

    The whole crop goes through the sampler, but everything past the ramp is pasted at zero weight, so
    marking the crop would report nine times the region at the default crop factor.
    """
    reach = _paste_band(box, crop, feather, width, height) * 2
    left, top, right, bottom = box
    crop_left, crop_top, crop_right, crop_bottom = crop
    return (max(crop_left, left - reach), max(crop_top, top - reach), min(crop_right, right + reach), min(crop_bottom, bottom + reach))


def _paint_box(mask: torch.Tensor, box: Box) -> torch.Tensor:
    left, top, right, bottom = box
    mask = mask.clone()
    mask[top:bottom, left:right] = 1.0
    return mask


def _detailer_info(
    *,
    active: bool,
    regions: int,
    skipped: list[int],
    region_size: int,
    max_size: int,
    crop_factor: float,
    feather: int,
    noise_feather: int,
    differential: bool,
    config,
    seed_mode: str,
    vary_per_region: bool,
    cycles: int,
    prompts: _RegionPrompts,
    loras_name: str,
) -> str:
    return format_info(
        "Detailer",
        {
            "Active": active,
            "Regions": regions,
            "RegionsSkipped": ", ".join(str(number) for number in skipped) if skipped else None,
            "RegionSize": region_size,
            "MaxSize": max_size,
            "CropFactor": round_float(crop_factor),
            "Feather": feather,
            "NoiseFeather": noise_feather,
            "Differential": differential,
            "RegionPositive": prompts.shared.positive if is_not_blank(prompts.shared.positive) else None,
            "RegionNegative": prompts.shared.negative if is_not_blank(prompts.shared.negative) else None,
            "RegionPositiveParts": _sections_info(prompts.positive),
            "RegionNegativeParts": _sections_info(prompts.negative),
            "Seed": config.seed,
            "SeedMode": seed_mode,
            "SeedPerRegion": vary_per_region,
            "Sampler": config.sampler,
            "Scheduler": config.scheduler,
            "Steps": config.steps,
            "Cfg": config.cfg,
            "Denoise": config.denoise,
            "Cycles": cycles if cycles > 1 else None,
            "Loras": f"\n{loras_name}" if is_not_blank(loras_name) else None,
        },
    )


def _sections_info(sections: RegionSections) -> str | None:
    if not sections.sections:
        return None

    return "\n" + "\n".join(f"{number}: {text}" for number, text in sorted(sections.sections.items()))
