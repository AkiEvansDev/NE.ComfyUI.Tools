from __future__ import annotations

from typing import Any

from ...core.base import PipelineInfo, append_info, round_float

__all__ = ["round_float", "parse_dimensions", "format_info", "append_info", "pipeline_info"]


def pipeline_info(info: PipelineInfo | None, block: str, name: str | None = None) -> PipelineInfo:
    """Extend the chain, tolerating an unconnected input so every step also works standalone."""
    base = info if isinstance(info, PipelineInfo) else PipelineInfo()
    return base.appended(block, name)


def parse_dimensions(dimensions: str) -> tuple[int, int]:
    width_part, height_part = [part.strip() for part in dimensions.split("x", 1)]
    width = int(width_part)
    height = int(height_part.split(" ", 1)[0])
    return width, height


def format_info(title: str, values: dict[str, Any]) -> str:
    lines = [f"[{title}]"]
    lines.extend(f"{key}: {value}" for key, value in values.items() if value is not None)
    return "\n".join(lines)
