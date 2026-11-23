"""Splitting a mask into the regions a detail pass works on.

`DetectRegions` and `DetailerStep` both call `mask_regions`, and that is the whole point of the module:
the number drawn on the detect preview has to be the number a `1: tags` section in the detailer prompt
addresses, and the only way to guarantee it is to derive both from one pass over the same mask. Detection
order cannot do it - boxes are found by score, they merge into one blob once they overlap, and by the time
the detailer sees a `MASK` there is nothing left of the boxes at all.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from ...constants import DETAILER_MASK_THRESHOLD, DETAILER_MAX_REGIONS, DETAILER_MIN_REGION_PIXELS

Box = tuple[int, int, int, int]


class Region(NamedTuple):
    """One blob of the mask: the box it fills, and its own pixels.

    The box alone is not enough to tell a region apart from its neighbours: two blobs whose boxes overlap -
    two hands of a seg mask, a hand-painted stroke around another - each hold part of the other inside
    their box, and a pass handed that box would repaint both. `blob` is this region's pixels only.
    """
    box: Box
    blob: torch.Tensor


def mask_regions(mask: torch.Tensor) -> list[Region]:
    """The mask's separate blobs, in reading order - top to bottom, then left to right.

    Each blob is detailed on its own: two hands in opposite corners share a bounding box that covers
    most of the frame, which would put them back at the resolution the detailer exists to escape.

    The cap keeps the largest blobs and the reading order is applied after it, so the position in this
    list is the region number - the one the preview draws and the one a prompt section names.
    """
    from scipy import ndimage

    labels, count = ndimage.label((mask >= DETAILER_MASK_THRESHOLD).cpu().numpy())
    if not count:
        return []

    found = []
    for label, bounds in enumerate(ndimage.find_objects(labels), start=1):
        rows, columns = bounds
        # This label only: another blob can sit inside the bounding box, and counting it too would let a
        # thin stroke around a large blob outrank it at the cap.
        own = labels[rows, columns] == label
        area = int(own.sum())
        if area < DETAILER_MIN_REGION_PIXELS:
            continue

        box = (int(columns.start), int(rows.start), int(columns.stop), int(rows.stop))
        blob = torch.zeros(mask.shape, dtype=torch.bool)
        blob[rows, columns] = torch.from_numpy(own)
        # Ordered by the blob rather than by any crop around it: the detect preview asks for no context and
        # the detailer for a lot of it, and a crop shifted off the frame edge would renumber both.
        found.append((area, (box[1], box[0]), Region(box, blob)))

    found.sort(key=lambda entry: entry[0], reverse=True)

    if len(found) > DETAILER_MAX_REGIONS:
        print(f"[comfyui-ne-tools] The mask holds {len(found)} regions, keeping the {DETAILER_MAX_REGIONS} largest.")

    return [region for _, _, region in sorted(found[:DETAILER_MAX_REGIONS], key=lambda entry: entry[1])]


def crop_box(box: Box, width: int, height: int, crop_factor: float) -> Box | None:
    """The box widened about its centre by `crop_factor` and aligned to 8, slid inside the frame.

    A multiple rather than a pixel count, because a fixed padding gives the same context to a 40px speck and
    to a 400px arm, which is the wrong way round for both. None only for a frame smaller than one cell.
    """
    left, top, right, bottom = box
    grow_x = int(round((right - left) * (crop_factor - 1.0) / 2.0))
    grow_y = int(round((bottom - top) * (crop_factor - 1.0) / 2.0))
    return _align_box(left - grow_x, top - grow_y, right + grow_x, bottom + grow_y, width, height)


def _align_box(left: int, top: int, right: int, bottom: int, width: int, height: int) -> Box | None:
    left = (left // 8) * 8
    top = (top // 8) * 8
    right = -(-right // 8) * 8
    bottom = -(-bottom // 8) * 8

    # Slid back into the frame rather than cut off at it. A hand in the corner would otherwise lose the
    # context on that side and be sampled at a composition the model never sees, which is the one place
    # a detail pass is already weakest.
    left, right = _slide(left, right, width)
    top, bottom = _slide(top, bottom, height)

    if right - left < 8 or bottom - top < 8:
        return None

    return (left, top, right, bottom)


def _slide(start: int, end: int, limit: int) -> tuple[int, int]:
    size = ((min(end - start, limit)) // 8) * 8

    if start < 0:
        start = 0
    elif start + size > limit:
        start = ((limit - size) // 8) * 8

    return start, start + size


def region_number(box: Box, regions: list[Region]) -> int | None:
    """The 1-based region a box ended up in, or None when the box did not survive the region limits.

    Overlapping boxes merge into one blob in the mask, so several of them share a number - which is
    exactly what the detailer will do with them.
    """
    best = 0
    number = None

    for index, region in enumerate(regions, start=1):
        overlap = _overlap(box, region.box)
        if overlap > best:
            best, number = overlap, index

    return number


def _overlap(box: Box, region: Box) -> int:
    width = min(box[2], region[2]) - max(box[0], region[0])
    height = min(box[3], region[3]) - max(box[1], region[1])
    return max(0, width) * max(0, height)
