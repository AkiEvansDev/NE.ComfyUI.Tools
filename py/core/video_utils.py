from __future__ import annotations

from nodes import CLIPVisionLoader


def load_clip_vision(clip_vision_path: str):
    result = CLIPVisionLoader().load_clip(clip_vision_path)
    return result[0] if isinstance(result, tuple) else result
