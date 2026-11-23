from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import folder_paths
import torch
from server import PromptServer

from ...constants import CATEGORY_ROOT, DISPLAY_FEEDBACK_EVENT
from ..core.base import any_type
from ..core.image import tensor2pil
from .helpers import Input
from .video_save import save_video_direct


DISPLAY_PREVIEW_PREFIX = "display_any"
# Channel counts an IMAGE can have, used to tell a (B, H, W) mask from an (H, W, C) frame.
IMAGE_CHANNELS = (1, 3, 4)


def serialize_display_value(value: Any) -> str:
    if value is None:
        return "None"

    if isinstance(value, str):
        return value

    if isinstance(value, bool | int | float):
        return str(value)

    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        try:
            return str(value)
        except Exception:
            return "Value exists, but could not be serialized."


def save_preview_image(image: torch.Tensor, unique_id: str) -> str:
    filename = f"{DISPLAY_PREVIEW_PREFIX}_{unique_id}.png"
    filepath = Path(folder_paths.get_temp_directory()) / filename
    tensor2pil(preview_frame(image)).save(filepath)

    return filename


def preview_frame(tensor: torch.Tensor) -> torch.Tensor:
    """The first frame of an IMAGE or a MASK, as an (H, W, 3) image tensor.

    Three dimensions are ambiguous — an unbatched image is (H, W, C) and a batched mask is (B, H, W) —
    so the last axis decides: a channel count on an image, a width on a mask.
    """
    if tensor.ndim == 4 or (tensor.ndim == 3 and tensor.shape[-1] not in IMAGE_CHANNELS):
        tensor = tensor[0]

    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)

    # A mask carries no channels of its own; PIL needs three to write anything but a bare greyscale.
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)

    return tensor


def save_preview_video(video: Any, unique_id: str) -> str | None:
    if not is_video_object(video):
        return None

    try:
        saved = save_video_direct(
            video,
            filename_prefix=f"{DISPLAY_PREVIEW_PREFIX}_{unique_id}",
            extension="auto",
            output_directory=folder_paths.get_temp_directory(),
        )
        return saved.file
    except Exception as error:
        print(f"Unable to save display video preview: {error}")
        return None


def is_image_tensor(value: Any) -> bool:
    # Two dimensions included: an unbatched MASK is (H, W), and it renders as well as any other.
    return isinstance(value, torch.Tensor) and value.ndim in (2, 3, 4)


def is_video_object(value: Any) -> bool:
    return hasattr(value, "save_to") and hasattr(value, "get_dimensions")


class DisplayAny:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={"source": (any_type, {})},
            hidden={"unique_id": "UNIQUE_ID"},
        )

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_ROOT

    def execute(self, source=None, unique_id=None):
        image = None
        video = None
        text = ""

        if is_image_tensor(source):
            image = save_preview_image(source, unique_id)
        elif is_video_object(source):
            video = save_preview_video(source, unique_id)
            text = "" if video else serialize_display_value(source)
        else:
            text = serialize_display_value(source)

        PromptServer.instance.send_sync(
            DISPLAY_FEEDBACK_EVENT,
            {
                "node": unique_id,
                "text": text,
                "image": image,
                "video": video,
            },
        )

        return {}
