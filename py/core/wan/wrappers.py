from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

import nodes
from nodes import VAEDecode

from ..base import VideoConfig


class WanNodeError(RuntimeError):
    pass


def _node_class(*names: str) -> type:
    for name in names:
        node_class = getattr(nodes, name, None)
        if node_class is not None:
            return node_class

    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
    for name in names:
        node_class = mappings.get(name)
        if node_class is not None:
            return node_class

    raise WanNodeError(
        "Missing ComfyUI Wan/video node. Tried: " + ", ".join(names)
    )


def _function_name(node: Any) -> str:
    return getattr(node, "FUNCTION", None) or "execute"


def _call_node(node_class: type, values: dict[str, Any], positional_fallback: Iterable[tuple[Any, ...]] = ()):  # noqa: ANN401
    node = node_class()
    function = getattr(node, _function_name(node))

    try:
        signature = inspect.signature(function)
        kwargs = {name: values[name] for name in signature.parameters if name in values}
        if kwargs:
            return function(**kwargs)
    except (TypeError, ValueError):
        pass

    for args in positional_fallback:
        try:
            return function(*args)
        except TypeError:
            continue

    node_name = getattr(node_class, "__name__", str(node_class))
    raise WanNodeError(f"Failed to call {node_name}: unsupported runtime signature.")


def _result_by_name(result: Any, node_class: type, name: str) -> Any:
    if not isinstance(result, tuple):
        result = (result,)

    names = getattr(node_class, "RETURN_NAMES", ()) or ()
    if name in names:
        index = names.index(name)
        if index < len(result):
            return result[index]

    for item in result:
        if isinstance(item, dict) and "samples" in item:
            return item

    return result[0] if result else None


def model_takes_clip_vision(model) -> bool:
    """Whether this Wan checkpoint has the img_emb projection that turns a CLIP Vision embedding into tokens."""
    # Wan 2.1 I2V has it, Wan 2.2 14B does not — and comfy/ldm/wan/model.py reserves 257 image tokens for an
    # embedding it was handed whether or not it can project them, so asking the model is the only safe test.
    # An unrecognisable object is treated as willing: that is the pre-2.2 behaviour, and it fails loudly.
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion_model is None:
        return True

    return getattr(diffusion_model, "img_emb", None) is not None


def clip_vision_encode(clip_vision, image):
    node_class = _node_class("CLIPVisionEncode")
    result = _call_node(
        node_class,
        {
            "clip_vision": clip_vision,
            "clip_vision_model": clip_vision,
            "image": image,
            "crop": "center",
        },
        positional_fallback=[
            (clip_vision, image, "center"),
            (clip_vision, image),
        ],
    )
    return result[0] if isinstance(result, tuple) else result


def apply_model_sampling_sd3(model, shift: float):
    node_class = _node_class("ModelSamplingSD3")
    result = _call_node(
        node_class,
        {
            "model": model,
            "shift": shift,
        },
        positional_fallback=[
            (model, shift),
        ],
    )
    return result[0] if isinstance(result, tuple) else result


def prepare_wan_video_latent(
    *,
    image,
    end_image,
    clip_vision_output,
    end_clip_vision_output,
    positive,
    negative,
    video_config: VideoConfig,
):
    if end_image is None:
        node_class = _node_class(
            "WanImageToVideo",
            "Wan22ImageToVideoLatent",
            "WanImageToVideoLatent",
        )
    else:
        node_class = _node_class(
            "WanFirstLastFrameToVideo",
            "WanFirstLastFrameToVideoLatent",
            "Wan22FirstLastFrameToVideoLatent",
        )

    values = {
        "image": image,
        "start_image": image,
        "first_image": image,
        "end_image": end_image,
        "last_image": end_image,
        "clip_vision_output": clip_vision_output,
        "start_clip_vision_output": clip_vision_output,
        "first_clip_vision_output": clip_vision_output,
        "end_clip_vision_output": end_clip_vision_output,
        "last_clip_vision_output": end_clip_vision_output,
        "positive": positive,
        "negative": negative,
        "width": video_config.width,
        "height": video_config.height,
        "length": video_config.frames,
        "frames": video_config.frames,
        "fps": video_config.fps,
        "batch_size": 1,
    }
    result = _call_node(node_class, values)

    if not isinstance(result, tuple):
        return result, positive, negative

    names = getattr(node_class, "RETURN_NAMES", ()) or ()
    latent = _result_by_name(result, node_class, "latent")
    if latent is None:
        latent = _result_by_name(result, node_class, "samples")

    if "positive" in names:
        positive = result[names.index("positive")]
    if "negative" in names:
        negative = result[names.index("negative")]

    return latent, positive, negative


def decode_video_latent(vae, latent):
    return VAEDecode().decode(vae, latent)


def _first_existing_result(result: Any, node_class: type, names: tuple[str, ...], index: int = 0) -> Any:
    if not isinstance(result, tuple):
        result = (result,)

    return_names = getattr(node_class, "RETURN_NAMES", ()) or ()
    for name in names:
        if name in return_names:
            result_index = return_names.index(name)
            if result_index < len(result):
                return result[result_index]

    return result[index] if index < len(result) else None


def prepare_wan_vace_control_video(
    *,
    image,
    end_image,
    video_config: VideoConfig,
    empty_frame_level: float,
):
    node_class = _node_class(
        "WanVideoVACEStartToEndFrame",
        "WanVACEStartToEndFrame",
        "WanVideoVaceStartToEndFrame",
    )
    values = {
        "start_image": image,
        "first_image": image,
        "image": image,
        "end_image": end_image,
        "last_image": end_image,
        "width": video_config.width,
        "height": video_config.height,
        "length": video_config.frames,
        "frames": video_config.frames,
        "empty_frame_level": empty_frame_level,
    }
    result = _call_node(node_class, values)
    control_video = _first_existing_result(result, node_class, ("control_video", "video", "images"), 0)
    mask = _first_existing_result(result, node_class, ("mask", "control_mask", "_mask"), 1)
    return control_video, mask


def prepare_wan_vace_video_latent(
    *,
    positive,
    negative,
    vae,
    control_video,
    mask,
    video_config: VideoConfig,
    strength: float,
):
    node_class = _node_class(
        "WanVaceToVideo",
        "WanVACEToVideo",
        "WanVideoVaceToVideo",
        "WanVideoVACEToVideo",
    )
    values = {
        "positive": positive,
        "negative": negative,
        "vae": vae,
        "control_video": control_video,
        "video": control_video,
        "control_mask": mask,
        "mask": mask,
        "_mask": mask,
        "width": video_config.width,
        "height": video_config.height,
        "length": video_config.frames,
        "frames": video_config.frames,
        "strength": strength,
    }
    result = _call_node(node_class, values)

    if not isinstance(result, tuple):
        return result, positive, negative

    latent = _first_existing_result(result, node_class, ("latent", "samples"), 0)
    names = getattr(node_class, "RETURN_NAMES", ()) or ()
    if "positive" in names:
        positive = result[names.index("positive")]
    if "negative" in names:
        negative = result[names.index("negative")]
    return latent, positive, negative
