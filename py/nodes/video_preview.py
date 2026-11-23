from __future__ import annotations

import folder_paths

from ...constants import CATEGORY_VIDEO
from .helpers import Input
from .image_save import sanitize_filename_part
from .video_save import save_video_direct


class VideoPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "video": ("VIDEO",),
            },
            optional={
                "info": ("PIPELINE_INFO",),
            },
        )

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = CATEGORY_VIDEO
    OUTPUT_NODE = True

    def preview(self, video, info=None):
        return preview_video(video, (info.name if info else None) or "video_preview")


def preview_video(video, filename_prefix: str = "video_preview"):
    from comfy_api.latest import io, ui

    safe_prefix = sanitize_filename_part(filename_prefix) or "video_preview"
    saved = save_video_direct(
        video,
        safe_prefix,
        "auto",
        output_directory=folder_paths.get_temp_directory(),
        delimiter="_",
        number_padding=4,
        number_start=False,
    )
    return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(saved.file, saved.subfolder, io.FolderType.temp)]))
