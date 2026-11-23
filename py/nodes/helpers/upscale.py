from __future__ import annotations

import torch
import torch.nn.functional as F

from nodes import ImageScale
import comfy.model_management
import comfy.utils

from ....constants import *
from ...core.base import ImageUpscaleConfig, VideoUpscaleConfig, extract_filename
from .config import upscale_model_choices
from .formatting import format_info, round_float
from .inputs import Input

# Tile geometry for the upscale models, and the stock node's own guess at how much a tile of that size
# costs in weights and activations. None of the three is a node input, so they stay module-private.
UPSCALE_TILE = 512
UPSCALE_TILE_OVERLAP = 32
UPSCALE_MEMORY_ESTIMATE = 384.0

_CPU_DEVICE = torch.device("cpu")


def _get_tiled_scale_steps(images, tile_x: int, tile_y: int, overlap: int) -> int:
    try:
        return comfy.utils.get_tiled_scale_steps(
            images.shape[3],
            images.shape[2],
            tile_x=tile_x,
            tile_y=tile_y,
            overlap=overlap,
        ) * images.shape[0]
    except Exception:
        return max(1, images.shape[0])


def _upscale_chunk(image, config: ImageUpscaleConfig, upscale_model, device, pbar, output_device):
    """One image or batch of frames through an already-resident upscale model.

    `output_device` is where the tiles are blended and the resize happens. tiled_scale defaults it to the
    CPU, and that is where the cost of an upscale actually lands: every tile crosses back, the feather mask
    is built there, and the accumulate-and-divide runs over the full-resolution result.
    """
    in_img = image.movedim(-1, -3).to(device)
    tile = UPSCALE_TILE
    overlap = UPSCALE_TILE_OVERLAP

    while True:
        try:
            # As float, the way the stock node calls it: under --fp16-intermediates the frames arrive as half.
            output = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a.float()),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                output_device=output_device,
                pbar=pbar,
            )
            break
        except comfy.model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    output = torch.clamp(output.movedim(-3, -1), min=0, max=1.0)
    return resize_image(output, config.upscale_method, config.resize)


def resize_image(image, upscale_method: str, resize: float):
    if resize == 1:
        return (image,)

    _, height, width, _ = image.shape
    return ImageScale().upscale(
        image,
        upscale_method,
        int(width * resize),
        int(height * resize),
        "disabled",
    )


def upscale_image_with_config(image, config: ImageUpscaleConfig):
    # A single image fits on the card whole — 1280x1920 at 4x is under half a gigabyte — so the tiles are
    # blended and the resize is done there, and only the finished image crosses back. The video path cannot
    # do this: 81 frames at that size are several gigabytes, which is why tiled_scale defaults to the CPU.
    upscale_model = config.model
    device = upscale_model.patcher.load_device
    pbar = _upscale_progress_bar(image)
    _reserve_upscale_memory(upscale_model, image, on_device_output=True)

    try:
        image, = _upscale_chunk(image, config, upscale_model, device, pbar, output_device=device)
    except comfy.model_management.OOM_EXCEPTION:
        # Halving the tile does not help when it is the output that will not fit, so fall back to the
        # stock behaviour of assembling on the CPU rather than failing the run.
        comfy.model_management.soft_empty_cache()
        image, = _upscale_chunk(image, config, upscale_model, device, pbar, output_device=_CPU_DEVICE)

    return (image.to(comfy.model_management.intermediate_device(), comfy.model_management.intermediate_dtype()),)


def _upscale_progress_bar(images):
    steps = _get_tiled_scale_steps(images.movedim(-1, -3), UPSCALE_TILE, UPSCALE_TILE, UPSCALE_TILE_OVERLAP)
    return comfy.utils.ProgressBar(steps) if steps > 0 else None


def _reserve_upscale_memory(upscale_model, image, *, on_device_output: bool) -> None:
    """Ask the model manager for room before loading, the way the stock upscale node does.

    `force_full_load` as upstream has it since 2026-08: without it a partial load under
    --disable-dynamic-vram leaves the weights split across devices and the model fails to run.
    """
    element = image.element_size()
    memory_required = (512 * 512 * 3) * element * max(upscale_model.scale, 1.0) * UPSCALE_MEMORY_ESTIMATE
    memory_required += image.nelement() * element

    if on_device_output:
        memory_required += image.nelement() * element * (upscale_model.scale ** 2)

    comfy.model_management.load_models_gpu([upscale_model.patcher], memory_required=memory_required, force_full_load=True)


def video_upscale_param_inputs() -> dict:
    return {
        "batch_size": Input.int(
            DEFAULT_VIDEO_UPSCALE_BATCH_SIZE,
            VIDEO_UPSCALE_BATCH_SIZE_MIN,
            VIDEO_UPSCALE_BATCH_SIZE_MAX,
            VIDEO_UPSCALE_BATCH_SIZE_STEP,
        ),
        "model": upscale_model_choices(),
        "upscale_method": Input.choice(IMAGE_RESIZE_METHODS, DEFAULT_IMAGE_RESIZE_METHOD),
        "resize": Input.float(
            DEFAULT_IMAGE_UPSCALE_RESIZE,
            IMAGE_UPSCALE_RESIZE_MIN,
            IMAGE_UPSCALE_RESIZE_MAX,
            IMAGE_UPSCALE_RESIZE_STEP,
        ),
        "temporal_smoothing_mode": Input.choice(VIDEO_TEMPORAL_SMOOTHING_MODES, DEFAULT_VIDEO_TEMPORAL_SMOOTHING_MODE),
        "temporal_smoothing": Input.float(
            DEFAULT_VIDEO_TEMPORAL_SMOOTHING,
            VIDEO_TEMPORAL_SMOOTHING_MIN,
            VIDEO_TEMPORAL_SMOOTHING_MAX,
            VIDEO_TEMPORAL_SMOOTHING_STEP,
        ),
        "temporal_sharpen": Input.float(
            DEFAULT_VIDEO_TEMPORAL_SHARPEN,
            VIDEO_TEMPORAL_SHARPEN_MIN,
            VIDEO_TEMPORAL_SHARPEN_MAX,
            VIDEO_TEMPORAL_SHARPEN_STEP,
        ),
        "motion_sensitivity": Input.float(
            DEFAULT_VIDEO_MOTION_SENSITIVITY,
            VIDEO_MOTION_SENSITIVITY_MIN,
            VIDEO_MOTION_SENSITIVITY_MAX,
            VIDEO_MOTION_SENSITIVITY_STEP,
        ),
        "edge_aware_smoothing": Input.bool(DEFAULT_VIDEO_EDGE_AWARE_SMOOTHING),
        "edge_preservation": Input.float(
            DEFAULT_VIDEO_EDGE_PRESERVATION,
            VIDEO_EDGE_PRESERVATION_MIN,
            VIDEO_EDGE_PRESERVATION_MAX,
            VIDEO_EDGE_PRESERVATION_STEP,
        ),
        "sharpen": Input.float(
            DEFAULT_VIDEO_SHARPEN,
            VIDEO_SHARPEN_MIN,
            VIDEO_SHARPEN_MAX,
            VIDEO_SHARPEN_STEP,
        ),
    }


def make_video_upscale_config(*, model: str, upscale_method: str, resize: float, **temporal) -> VideoUpscaleConfig:
    """Turn the flat widgets of video_upscale_param_inputs() into the nested config.

    The remaining widget names are the temporal field names verbatim, so they flow straight through;
    rounding and range checks belong to the dataclass, not to the callers.
    """
    image_upscale = ImageUpscaleConfig(model_name=model, upscale_method=upscale_method, resize=resize)
    return VideoUpscaleConfig(image_upscale=image_upscale, **temporal)


def _edge_mask(images):
    x = images.permute(0, 3, 1, 2).float()
    channels = x.shape[1]
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=channels)
    gy = F.conv2d(x, ky, padding=1, groups=channels)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-12).mean(dim=1, keepdim=True)
    edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return edge.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _apply_edge_aware_strength(current, strength: float, edge_aware: bool, edge_preservation: float):
    if not edge_aware or edge_preservation <= 0:
        return strength
    edge_preservation = max(
        VIDEO_EDGE_PRESERVATION_MIN,
        min(VIDEO_EDGE_PRESERVATION_MAX, round_float(edge_preservation)),
    )
    return strength * (1.0 - _edge_mask(current) * edge_preservation)


def temporal_smooth_images(images, strength: float, edge_aware: bool = False, edge_preservation: float = DEFAULT_VIDEO_EDGE_PRESERVATION):
    strength = round_float(strength)
    if strength <= 0 or images.shape[0] < 2:
        return images

    strength = max(VIDEO_TEMPORAL_SMOOTHING_MIN, min(VIDEO_TEMPORAL_SMOOTHING_MAX, strength))
    current = images.float()
    previous = torch.cat((current[:1], current[:-1]), dim=0)
    next_frame = torch.cat((current[1:], current[-1:]), dim=0)
    neighbors = (previous + next_frame) * 0.5
    strength_map = _apply_edge_aware_strength(current, strength, edge_aware, edge_preservation)
    smoothed = current * (1.0 - strength_map) + neighbors * strength_map
    return smoothed.clamp(0.0, 1.0).to(dtype=images.dtype)


def temporal_smooth_images_adaptive(
    images,
    strength: float,
    motion_sensitivity: float = DEFAULT_VIDEO_MOTION_SENSITIVITY,
    edge_aware: bool = False,
    edge_preservation: float = DEFAULT_VIDEO_EDGE_PRESERVATION,
):
    strength = round_float(strength)
    if strength <= 0 or images.shape[0] < 2:
        return images

    strength = max(VIDEO_TEMPORAL_SMOOTHING_MIN, min(VIDEO_TEMPORAL_SMOOTHING_MAX, strength))
    motion_sensitivity = max(
        VIDEO_MOTION_SENSITIVITY_MIN,
        min(VIDEO_MOTION_SENSITIVITY_MAX, round_float(motion_sensitivity)),
    )

    current = images.float()
    previous = torch.cat((current[:1], current[:-1]), dim=0)
    next_frame = torch.cat((current[1:], current[-1:]), dim=0)

    diff_prev = (current - previous).abs().mean(dim=(1, 2, 3), keepdim=True)
    diff_next = (current - next_frame).abs().mean(dim=(1, 2, 3), keepdim=True)

    weight_prev = torch.exp(-diff_prev * motion_sensitivity)
    weight_next = torch.exp(-diff_next * motion_sensitivity)
    neighbors = (previous * weight_prev + next_frame * weight_next) / (weight_prev + weight_next + 1e-6)
    strength_map = _apply_edge_aware_strength(current, strength, edge_aware, edge_preservation)
    smoothed = current * (1.0 - strength_map) + neighbors * strength_map
    return smoothed.clamp(0.0, 1.0).to(dtype=images.dtype)


def temporal_smooth_images_with_config(images, config: VideoUpscaleConfig):
    if config.temporal_smoothing_mode == "off":
        return images
    if config.temporal_smoothing_mode == "adaptive":
        return temporal_smooth_images_adaptive(
            images,
            config.temporal_smoothing,
            config.motion_sensitivity,
            config.edge_aware_smoothing,
            config.edge_preservation,
        )
    return temporal_smooth_images(
        images,
        config.temporal_smoothing,
        config.edge_aware_smoothing,
        config.edge_preservation,
    )


def sharpen_images(images, amount: float):
    amount = round_float(amount)
    if amount <= 0:
        return images

    amount = max(VIDEO_SHARPEN_MIN, min(VIDEO_SHARPEN_MAX, amount))
    x = images.permute(0, 3, 1, 2).float()
    blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    sharpened = x + (x - blur) * amount
    return sharpened.clamp(0.0, 1.0).permute(0, 2, 3, 1).to(dtype=images.dtype)


def temporal_sharpen_images(images, amount: float, motion_sensitivity: float = DEFAULT_VIDEO_MOTION_SENSITIVITY):
    amount = round_float(amount)
    if amount <= 0 or images.shape[0] < 2:
        return images

    amount = max(VIDEO_TEMPORAL_SHARPEN_MIN, min(VIDEO_TEMPORAL_SHARPEN_MAX, amount))
    motion_sensitivity = max(
        VIDEO_MOTION_SENSITIVITY_MIN,
        min(VIDEO_MOTION_SENSITIVITY_MAX, round_float(motion_sensitivity)),
    )

    current = images.float()
    previous = torch.cat((current[:1], current[:-1]), dim=0)
    next_frame = torch.cat((current[1:], current[-1:]), dim=0)
    neighbors = (previous + next_frame) * 0.5
    stability = torch.exp(-(current - neighbors).abs().mean(dim=3, keepdim=True) * motion_sensitivity)

    x = current.permute(0, 3, 1, 2)
    blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    detail = (x - blur).permute(0, 2, 3, 1)
    sharpened = current + detail * stability * amount
    return sharpened.clamp(0.0, 1.0).to(dtype=images.dtype)


def video_upscale_info(*, active: bool, config: VideoUpscaleConfig) -> str:
    return format_info(
        "Video Upscale",
        {
            "Active": active,
            "BatchSize": config.batch_size,
            "Model": extract_filename(config.model_name),
            "Method": config.upscale_method,
            "Resize": config.resize,
            "SmoothingMode": config.temporal_smoothing_mode,
            "TemporalSmoothing": config.temporal_smoothing,
            "TemporalSharpen": config.temporal_sharpen,
            "MotionSensitivity": config.motion_sensitivity,
            "EdgeAwareSmoothing": config.edge_aware_smoothing,
            "EdgePreservation": config.edge_preservation,
            "Sharpen": config.sharpen,
        },
    )


def upscale_video_images_with_config(images, config: VideoUpscaleConfig):
    batch_size = max(1, int(config.batch_size))
    chunks = []
    total_steps = 0
    for index in range(0, images.shape[0], batch_size):
        chunk = images[index:index + batch_size]
        total_steps += _get_tiled_scale_steps(chunk.movedim(-1, -3), UPSCALE_TILE, UPSCALE_TILE, UPSCALE_TILE_OVERLAP)

    pbar = comfy.utils.ProgressBar(total_steps) if total_steps > 0 else None

    upscale_model = config.image_upscale.model
    device = upscale_model.patcher.load_device

    # Loaded once for the whole clip, and through the model manager rather than with a bare `.to(device)`: the
    # manager is the one that knows the Wan experts are still resident and can evict them to make room. Moved
    # by hand, the upscaler squeezed in beside them and the batch fell back to assembling on the CPU.
    _reserve_upscale_memory(upscale_model, images[:batch_size], on_device_output=True)

    # Tiles are blended wherever the output lives, and that is where the processor time goes: eight frames
    # assembled on the CPU measured 32 s of it against 3 s on the card, for the same wall clock. A whole
    # clip at output resolution does not fit there, but one batch does — so each is assembled on the card
    # and copied down.
    output_device = device
    try:
        for index in range(0, images.shape[0], batch_size):
            batch = images[index:index + batch_size]

            try:
                chunk, = _upscale_chunk(batch, config.image_upscale, upscale_model, device, pbar, output_device)
            except comfy.model_management.OOM_EXCEPTION:
                # Stay down for the rest of the clip: retrying per batch would pay the failure every time.
                comfy.model_management.soft_empty_cache()
                output_device = _CPU_DEVICE
                chunk, = _upscale_chunk(batch, config.image_upscale, upscale_model, device, pbar, output_device)

            chunks.append(chunk.to(_CPU_DEVICE))
    finally:
        comfy.model_management.soft_empty_cache()

    images = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    images = _in_frame_chunks(images, batch_size, 1, lambda part: temporal_smooth_images_with_config(part, config))
    images = _in_frame_chunks(
        images, batch_size, 1, lambda part: temporal_sharpen_images(part, config.temporal_sharpen, config.motion_sensitivity),
    )
    return _in_frame_chunks(images, batch_size, 0, lambda part: sharpen_images(part, config.sharpen))


def _in_frame_chunks(images, chunk_size: int, halo: int, apply):
    """Run a post-processing step over slices of the clip rather than over all of it at once.

    A whole-clip pass peaked at nine copies of the video — some 13 GB for an 81-frame render, which is
    allocator traffic rather than arithmetic. Every step here reads at most `t±1` and normalises per frame
    (`_edge_mask` takes its maximum over one frame's own pixels), so a slice that carries `halo` neighbours
    on each side and then drops them produces exactly what the whole clip would.
    """
    frames = images.shape[0]
    if chunk_size < 1 or frames <= chunk_size:
        return apply(images)

    parts = []
    for start in range(0, frames, chunk_size):
        stop = min(start + chunk_size, frames)
        low = max(0, start - halo)
        high = min(frames, stop + halo)
        parts.append(apply(images[low:high])[start - low:stop - low])

    return torch.cat(parts, dim=0)
