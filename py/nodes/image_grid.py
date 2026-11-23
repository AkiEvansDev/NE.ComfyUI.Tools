from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from ...constants import (
    CATEGORY_IMAGE,
    DEFAULT_GRID_CELL_GAP,
    DEFAULT_GRID_CLEAR_ON_COMPLETE,
    DEFAULT_GRID_CLEAR_ON_FIRST_CELL,
    DEFAULT_GRID_EMPTY_CELLS,
    DEFAULT_GRID_LABEL_SIZE,
    DEFAULT_GRID_PADDING,
    DEFAULT_GRID_SHOW_PARTIAL,
    DEFAULT_GRID_THEME,
    DEFAULT_GRID_Z_GAP,
    GRID_GAP_MAX,
    GRID_GAP_MIN,
    GRID_LABEL_FONTS,
    GRID_LABEL_SIZE_MAX,
    GRID_LABEL_SIZE_MIN,
    GRID_THEMES,
    GRID_Z_GAP_MAX,
)
from ..core.base import XYRangeConfig, XYZRangeConfig
from ..core.image import pil2tensor, tensor2pil
from ..server.state import xy_grid_registry
from .helpers import Input

_THEME_COLORS = {
    "light": {
        "background": (255, 255, 255),
        "tile": (255, 255, 255),
        "text": (20, 20, 20),
        "empty_outline": (210, 210, 210),
        "empty_cross": (235, 235, 235),
    },
    "dark": {
        "background": (24, 24, 24),
        "tile": (18, 18, 20),
        "text": (245, 245, 245),
        "empty_outline": (82, 82, 91),
        "empty_cross": (63, 63, 70),
    },
}


@dataclass
class _StoredImage:
    image: Image.Image | None = None
    path: str | None = None

    def load(self) -> Image.Image:
        if self.image is not None:
            return self.image

        if self.path is None:
            raise ValueError("Stored image has neither PIL data nor a path")

        with Image.open(self.path) as image:
            return image.convert("RGB").copy()

    def cleanup(self) -> None:
        if self.path:
            Path(self.path).unlink(missing_ok=True)
            self.path = None


@dataclass
class _XYGridState:
    signature: tuple[int, int, int, int]
    images: dict[tuple[int, int], _StoredImage] = field(default_factory=dict)
    x_labels: dict[int, str] = field(default_factory=dict)
    y_labels: dict[int, str] = field(default_factory=dict)

    def cleanup(self) -> None:
        for stored in self.images.values():
            stored.cleanup()


@dataclass
class _XYZGridState:
    signature: tuple[int, int, int, int, int, int]
    images: dict[tuple[int, int, int], _StoredImage] = field(default_factory=dict)
    x_labels: dict[int, str] = field(default_factory=dict)
    y_labels: dict[int, str] = field(default_factory=dict)
    z_labels: dict[int, str] = field(default_factory=dict)

    def cleanup(self) -> None:
        for stored in self.images.values():
            stored.cleanup()


class XYImageGrid:
    def __init__(self):
        self._unique_id: str | None = None

    def __del__(self):
        if self._unique_id is None:
            return

        prefix = self._registry_prefix(self._unique_id)
        for key in list(xy_grid_registry.keys()):
            if key.startswith(prefix):
                self._pop_state(key)

    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "image": ("IMAGE",),
                "config": ("XY_RANGE_CONFIG",),
                "x_label": Input.string(force_input=True),
                "y_label": Input.string(force_input=True),
                "theme": Input.choice(GRID_THEMES, DEFAULT_GRID_THEME),
                "show_partial": Input.bool(DEFAULT_GRID_SHOW_PARTIAL),
                "clear_on_complete": Input.bool(DEFAULT_GRID_CLEAR_ON_COMPLETE),
                "clear_on_first_cell": Input.bool(DEFAULT_GRID_CLEAR_ON_FIRST_CELL),
                "empty_cells": Input.bool(DEFAULT_GRID_EMPTY_CELLS),
                "cell_gap": Input.int(DEFAULT_GRID_CELL_GAP, GRID_GAP_MIN, GRID_GAP_MAX),
                "padding": Input.int(DEFAULT_GRID_PADDING, GRID_GAP_MIN, GRID_GAP_MAX),
                "label_size": Input.int(DEFAULT_GRID_LABEL_SIZE, GRID_LABEL_SIZE_MIN, GRID_LABEL_SIZE_MAX),
            },
            optional={
                "title": Input.string(force_input=True),
            },
            hidden={"unique_id": "UNIQUE_ID"},
        )

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE

    def execute(
        self,
        image: torch.Tensor,
        config: XYRangeConfig,
        x_label: str,
        y_label: str,
        theme: str,
        show_partial: bool,
        clear_on_complete: bool,
        clear_on_first_cell: bool,
        empty_cells: bool,
        cell_gap: int,
        padding: int,
        label_size: int,
        title: str | None = None,
        unique_id: str | None = None,
    ):
        self._unique_id = unique_id

        key = self._registry_key(unique_id, config)
        signature = config.signature

        if clear_on_first_cell and config.first_cell:
            self._pop_state(key)

        state = xy_grid_registry.get(key)
        if state is None or state.signature != signature:
            self._cleanup_state(state)
            state = _XYGridState(signature=signature)
            xy_grid_registry[key] = state

        pil_image = tensor2pil(image[0]).convert("RGB")

        state.images[(config.x, config.y)] = self._store_image(
            state.images.get((config.x, config.y)),
            pil_image,
            keep_in_memory=show_partial,
        )
        state.x_labels[config.x] = self._label(x_label, config.x)
        state.y_labels[config.y] = self._label(y_label, config.y)

        collected = len(state.images)
        complete = collected >= config.total

        if complete or show_partial:
            result = self._compose_grid(
                state=state,
                config=config,
                cell_gap=cell_gap,
                padding=padding,
                label_size=label_size,
                empty_cells=empty_cells,
                theme=theme,
                title=title,
            )
        else:
            result = pil_image

        info = f"XY grid: {collected}/{config.total}"

        if complete:
            info = "XY grid complete"

            if clear_on_complete:
                self._pop_state(key)

        return (pil2tensor(result), info)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def _registry_prefix(self, unique_id: str) -> str:
        return f"{unique_id}:"

    def _registry_key(self, unique_id: str | None, config: XYRangeConfig) -> str:
        node_id = unique_id or "global"
        return f"{node_id}:xy:{config.x_start}:{config.x_end}:{config.y_start}:{config.y_end}"

    def _pop_state(self, key: str) -> None:
        state = xy_grid_registry.pop(key, None)
        self._cleanup_state(state)

    def _cleanup_state(self, state) -> None:
        if hasattr(state, "cleanup"):
            state.cleanup()

    def _store_image(self, previous: _StoredImage | None, image: Image.Image, keep_in_memory: bool) -> _StoredImage:
        if previous is not None:
            previous.cleanup()

        if keep_in_memory:
            return _StoredImage(image=image.copy())

        handle = tempfile.NamedTemporaryFile(prefix="ne_xy_grid_", suffix=".png", delete=False)
        path = handle.name
        handle.close()
        image.save(path)
        return _StoredImage(path=path)

    def _label(self, value: str | None, fallback: int) -> str:
        value = value.strip() if value else ""
        return value if value else str(fallback)

    def _font(self, size: int):
        for path in GRID_LABEL_FONTS:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass

        return ImageFont.load_default()

    def _text_size(self, draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
        if not text:
            return 0, 0

        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _line_height(self, draw: ImageDraw.ImageDraw, font) -> int:
        return self._text_size(draw, "Ag", font)[1]

    def _wrap_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font,
        max_width: int,
        max_lines: int | None = None,
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        words = []
        for part in text.splitlines():
            words.extend(part.split() or [part])

        lines: list[str] = []
        current = ""

        for word in words:
            candidate = f"{current} {word}".strip()
            if not current or self._text_size(draw, candidate, font)[0] <= max_width:
                current = candidate
                continue

            lines.append(current)
            current = word

        if current:
            lines.append(current)

        fitted: list[str] = []
        for line in lines:
            if self._text_size(draw, line, font)[0] <= max_width:
                fitted.append(line)
                continue

            chunk = ""
            for char in line:
                candidate = chunk + char
                if chunk and self._text_size(draw, candidate, font)[0] > max_width:
                    fitted.append(chunk)
                    chunk = char
                else:
                    chunk = candidate
            if chunk:
                fitted.append(chunk)

        if max_lines is not None and len(fitted) > max_lines:
            fitted = fitted[:max_lines]
            last = fitted[-1]
            ellipsis = "..."
            while last and self._text_size(draw, last + ellipsis, font)[0] > max_width:
                last = last[:-1]
            fitted[-1] = f"{last}{ellipsis}" if last else ellipsis

        return "\n".join(fitted)

    def _compose_grid(
        self,
        state: _XYGridState,
        config: XYRangeConfig,
        cell_gap: int,
        padding: int,
        label_size: int,
        empty_cells: bool,
        theme: str,
        title: str | None,
    ) -> Image.Image:
        loaded_images = {key: stored.load() for key, stored in state.images.items()}
        images = list(loaded_images.values())
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)

        x_values = list(range(config.x_start, config.x_end + 1))
        y_values = list(range(config.y_start, config.y_end + 1))

        colors = _THEME_COLORS.get(theme, _THEME_COLORS[DEFAULT_GRID_THEME])
        font = self._font(label_size)
        title_font = self._font(label_size + 6)

        measure = Image.new("RGB", (1, 1), colors["background"])
        draw = ImageDraw.Draw(measure)

        max_x_label_width = max(24, cell_width)
        x_labels = {
            x_value: self._wrap_text(
                draw,
                state.x_labels.get(x_value, str(x_value)),
                font,
                max_x_label_width,
                max_lines=3,
            )
            for x_value in x_values
        }
        x_label_height = max(
            [self._text_size(draw, label, font)[1] for label in x_labels.values()] + [self._line_height(draw, font)]
        ) + 12

        max_y_label_width = max(96, min(320, max(cell_width, label_size * 10)))
        y_labels = {
            y_value: self._wrap_text(
                draw,
                state.y_labels.get(y_value, str(y_value)),
                font,
                max_y_label_width,
                max_lines=4,
            )
            for y_value in y_values
        }
        y_label_width = max(
            [self._text_size(draw, label, font)[0] for label in y_labels.values()] + [48]
        ) + 16

        grid_width = config.x_count * cell_width + max(0, config.x_count - 1) * cell_gap
        content_width = y_label_width + grid_width
        width = padding * 2 + content_width

        title_text = ""
        title_height = 0
        if title and title.strip():
            title_text = self._wrap_text(draw, title.strip(), title_font, width - padding * 2, max_lines=3)
            title_height = self._text_size(draw, title_text, title_font)[1] + 20

        height = (
            padding * 2
            + title_height
            + x_label_height
            + config.y_count * cell_height
            + max(0, config.y_count - 1) * cell_gap
        )

        canvas = Image.new("RGB", (width, height), colors["background"])
        draw = ImageDraw.Draw(canvas)

        top = padding

        if title_text:
            title_width, _ = self._text_size(draw, title_text, title_font)
            draw.multiline_text(
                ((width - title_width) // 2, top),
                title_text,
                fill=colors["text"],
                font=title_font,
                spacing=6,
                align="center",
            )
            top += title_height

        image_left = padding + y_label_width
        image_top = top + x_label_height

        for column, x_value in enumerate(x_values):
            left = image_left + column * (cell_width + cell_gap)
            label = x_labels[x_value]
            label_width, label_height = self._text_size(draw, label, font)
            draw.multiline_text(
                (left + (cell_width - label_width) // 2, top + (x_label_height - label_height) // 2),
                label,
                fill=colors["text"],
                font=font,
                spacing=4,
                align="center",
            )

        for row, y_value in enumerate(y_values):
            cell_top = image_top + row * (cell_height + cell_gap)

            y_label = y_labels[y_value]
            _, y_label_height = self._text_size(draw, y_label, font)
            draw.multiline_text(
                (padding, cell_top + (cell_height - y_label_height) // 2),
                y_label,
                fill=colors["text"],
                font=font,
                spacing=4,
            )

            for column, x_value in enumerate(x_values):
                cell_left = image_left + column * (cell_width + cell_gap)
                cell = loaded_images.get((x_value, y_value))

                if cell is None:
                    if empty_cells:
                        self._draw_empty_cell(
                            draw,
                            cell_left,
                            cell_top,
                            cell_width,
                            cell_height,
                            colors,
                        )
                    continue

                tile = Image.new("RGB", (cell_width, cell_height), colors["tile"])
                tile.paste(cell, ((cell_width - cell.width) // 2, (cell_height - cell.height) // 2))
                canvas.paste(tile, (cell_left, cell_top))

        return canvas

    def _draw_empty_cell(
        self,
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        width: int,
        height: int,
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        draw.rectangle(
            (left, top, left + width - 1, top + height - 1),
            outline=colors["empty_outline"],
            width=2,
        )
        draw.line(
            (left, top, left + width - 1, top + height - 1),
            fill=colors["empty_cross"],
            width=2,
        )
        draw.line(
            (left + width - 1, top, left, top + height - 1),
            fill=colors["empty_cross"],
            width=2,
        )


class XYZImageGrid(XYImageGrid):
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "image": ("IMAGE",),
                "config": ("XYZ_RANGE_CONFIG",),
                "x_label": Input.string(force_input=True),
                "y_label": Input.string(force_input=True),
                "z_label": Input.string(force_input=True),
                "theme": Input.choice(GRID_THEMES, DEFAULT_GRID_THEME),
                "show_partial": Input.bool(DEFAULT_GRID_SHOW_PARTIAL),
                "clear_on_complete": Input.bool(DEFAULT_GRID_CLEAR_ON_COMPLETE),
                "clear_on_first_cell": Input.bool(DEFAULT_GRID_CLEAR_ON_FIRST_CELL),
                "empty_cells": Input.bool(DEFAULT_GRID_EMPTY_CELLS),
                "cell_gap": Input.int(DEFAULT_GRID_CELL_GAP, GRID_GAP_MIN, GRID_GAP_MAX),
                "padding": Input.int(DEFAULT_GRID_PADDING, GRID_GAP_MIN, GRID_GAP_MAX),
                "label_size": Input.int(DEFAULT_GRID_LABEL_SIZE, GRID_LABEL_SIZE_MIN, GRID_LABEL_SIZE_MAX),
                "z_grid_gap": Input.int(DEFAULT_GRID_Z_GAP, GRID_GAP_MIN, GRID_Z_GAP_MAX),
            },
            optional={
                "title": Input.string(force_input=True),
            },
            hidden={"unique_id": "UNIQUE_ID"},
        )

    def execute(
        self,
        image: torch.Tensor,
        config: XYZRangeConfig,
        x_label: str,
        y_label: str,
        z_label: str,
        theme: str,
        show_partial: bool,
        clear_on_complete: bool,
        clear_on_first_cell: bool,
        empty_cells: bool,
        cell_gap: int,
        padding: int,
        label_size: int,
        z_grid_gap: int,
        title: str | None = None,
        unique_id: str | None = None,
    ):
        self._unique_id = unique_id

        key = self._xyz_registry_key(unique_id, config)
        signature = config.signature

        if clear_on_first_cell and config.first_cell:
            self._pop_state(key)

        state = xy_grid_registry.get(key)
        if state is None or state.signature != signature:
            self._cleanup_state(state)
            state = _XYZGridState(signature=signature)
            xy_grid_registry[key] = state

        pil_image = tensor2pil(image[0]).convert("RGB")
        cell_key = (config.x, config.y, config.z)
        state.images[cell_key] = self._store_image(
            state.images.get(cell_key),
            pil_image,
            keep_in_memory=show_partial,
        )
        state.x_labels[config.x] = self._label(x_label, config.x)
        state.y_labels[config.y] = self._label(y_label, config.y)
        state.z_labels[config.z] = self._label(z_label, config.z)

        collected = len(state.images)
        complete = collected >= config.total

        if complete or show_partial:
            result = self._compose_xyz_grid(
                state=state,
                config=config,
                cell_gap=cell_gap,
                padding=padding,
                label_size=label_size,
                empty_cells=empty_cells,
                theme=theme,
                title=title,
                z_grid_gap=z_grid_gap,
            )
        else:
            result = pil_image

        info = f"XYZ grid: {collected}/{config.total}"

        if complete:
            info = "XYZ grid complete"

            if clear_on_complete:
                self._pop_state(key)

        return (pil2tensor(result), info)

    def _xyz_registry_key(self, unique_id: str | None, config: XYZRangeConfig) -> str:
        node_id = unique_id or "global"
        return (
            f"{node_id}:xyz:{config.x_start}:{config.x_end}:"
            f"{config.y_start}:{config.y_end}:{config.z_start}:{config.z_end}"
        )

    def _compose_xyz_grid(
        self,
        state: _XYZGridState,
        config: XYZRangeConfig,
        cell_gap: int,
        padding: int,
        label_size: int,
        empty_cells: bool,
        theme: str,
        title: str | None,
        z_grid_gap: int,
    ) -> Image.Image:
        z_values = list(range(config.z_start, config.z_end + 1))
        xy_images: list[Image.Image] = []

        for z_value in z_values:
            xy_state = _XYGridState(signature=(config.x_start, config.x_end, config.y_start, config.y_end))
            xy_state.x_labels = state.x_labels
            xy_state.y_labels = state.y_labels
            xy_state.images = {
                (x, y): stored
                for (x, y, z), stored in state.images.items()
                if z == z_value
            }

            if not xy_state.images:
                continue

            z_title = state.z_labels.get(z_value, str(z_value))
            xy_images.append(self._compose_grid(
                state=xy_state,
                config=config,
                cell_gap=cell_gap,
                padding=padding,
                label_size=label_size,
                empty_cells=empty_cells,
                theme=theme,
                title=z_title,
            ))

        if not xy_images:
            raise ValueError("XYZ grid has no images to compose")

        colors = _THEME_COLORS.get(theme, _THEME_COLORS[DEFAULT_GRID_THEME])
        tile_width = max(image.width for image in xy_images)
        tile_height = max(image.height for image in xy_images)
        count = len(xy_images)
        columns = max(1, math.ceil(math.sqrt(count * tile_height / max(1, tile_width))))
        rows = math.ceil(count / columns)

        grid_width = columns * tile_width + max(0, columns - 1) * z_grid_gap
        grid_height = rows * tile_height + max(0, rows - 1) * z_grid_gap

        title_text = ""
        title_height = 0
        font = self._font(label_size + 6)
        measure = Image.new("RGB", (1, 1), colors["background"])
        draw = ImageDraw.Draw(measure)
        if title and title.strip():
            title_text = self._wrap_text(draw, title.strip(), font, grid_width, max_lines=3)
            title_height = self._text_size(draw, title_text, font)[1] + 20

        width = padding * 2 + grid_width
        height = padding * 2 + title_height + grid_height
        canvas = Image.new("RGB", (width, height), colors["background"])
        draw = ImageDraw.Draw(canvas)

        top = padding
        if title_text:
            title_width, _ = self._text_size(draw, title_text, font)
            draw.multiline_text(
                ((width - title_width) // 2, top),
                title_text,
                fill=colors["text"],
                font=font,
                spacing=6,
                align="center",
            )
            top += title_height

        for index, image in enumerate(xy_images):
            row = index // columns
            column = index % columns
            left = padding + column * (tile_width + z_grid_gap)
            tile_top = top + row * (tile_height + z_grid_gap)
            tile = Image.new("RGB", (tile_width, tile_height), colors["background"])
            tile.paste(image, ((tile_width - image.width) // 2, (tile_height - image.height) // 2))
            canvas.paste(tile, (left, tile_top))

        return canvas
