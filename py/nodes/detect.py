from __future__ import annotations

import torch
from PIL import ImageDraw, ImageFont

from ...constants import *
from ..core.detect import (
    detect_model_names,
    detect_model_path,
    detect_objects,
    detections_to_mask,
    grow_detection,
    load_session,
)
from ..core.image import ensure_batch, pil2tensor, to_pil_rgb
from ..core.segment import outline_detections, segmenter_available
from ..core.regions import Box, mask_regions, region_number
from .helpers import Input, decoded_image, format_info, pipeline_info, round_float


class DetectRegions:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "active": Input.bool(True),
                "model": Input.choice(detect_model_names()),
                "confidence": Input.float(
                    DEFAULT_DETECT_CONFIDENCE,
                    DETECT_CONFIDENCE_MIN,
                    DETECT_CONFIDENCE_MAX,
                    DETECT_CONFIDENCE_STEP,
                ),
                "iou": Input.float(DEFAULT_DETECT_IOU, DETECT_IOU_MIN, DETECT_IOU_MAX, DETECT_IOU_STEP),
                "dilation": Input.int(DEFAULT_DETECT_DILATION, DETECT_DILATION_MIN, DETECT_DILATION_MAX, DETECT_DILATION_STEP),
                # Last, because saved workflows restore widget values by position.
                "shape": Input.choice(DETECT_SHAPES, DEFAULT_DETECT_SHAPE),
            },
            # Same pair of sources as UpscaleStep: pixels straight from a detailer, or a latent to decode.
            optional={
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "info": ("PIPELINE_INFO",),
            },
        )

    # The image passes straight through as the first output so the node can sit inline in a pipeline.
    RETURN_TYPES = ("IMAGE", "MASK", "PIPELINE_INFO", "IMAGE")
    RETURN_NAMES = ("image", "mask", "info", "preview")
    FUNCTION = "detect"
    CATEGORY = CATEGORY_DETECT

    def detect(
        self, active, model: str, confidence: float, iou: float, dilation: int, shape: str = DEFAULT_DETECT_SHAPE,
        image=None, latent=None, vae=None, info=None,
    ):
        image = ensure_batch(decoded_image(image, latent, vae))

        masks = []
        scores = []
        previews = []
        found_regions = 0
        drawn = False
        boxes_only = False
        # A box-only model gets its outlines from the segmenter, when one is installed.
        outline = shape == DETECT_SHAPE_SEGMENT and segmenter_available()

        for frame in image:
            if active:
                # Thresholds go in unrounded: rounding to two digits would turn a small one into zero.
                found = detect_objects(frame, load_session(detect_model_path(model)), confidence, iou)
                if outline:
                    found = outline_detections(frame, found)
            else:
                found = []

            # Dilation is applied once, here, so the preview shows the boxes the detailer will actually get.
            grown = [grow_detection(detection, dilation, frame.shape[1], frame.shape[0]) for detection in found]
            grown = [detection for detection in grown if detection is not None]

            drawn = drawn or bool(grown)
            scores.extend(detection.score for detection in grown)
            boxes_only = boxes_only or any(detection.mask is None for detection in grown)

            mask = detections_to_mask(grown, frame.shape[1], frame.shape[0], shape)
            # The numbers come out of the mask, not out of this list: overlapping boxes merge into one blob
            # and the detailer sees only the mask, so this is the one place both nodes can agree on.
            regions = mask_regions(mask[0]) if grown else []
            found_regions += len(regions)

            masks.append(mask)
            previews.append(_draw_boxes(frame, [(kept.box, kept.score, region_number(kept.box, regions)) for kept in grown]))

        # No box anywhere - switched off, or simply nothing found - means there is no preview to make,
        # and the image itself goes out as one. Drawing it through PIL would only copy the frame back.
        preview = torch.cat(previews, dim=0) if drawn else image

        block = format_info(
            "Detect",
            {
                "Active": active,
                "Model": model,
                "Found": len(scores),
                "Regions": found_regions,
                "Confidence": round_float(confidence),
                "Dilation": f"{dilation}%",
                "Shape": _shape_info(shape, outline, boxes_only),
                "Scores": _score_summary(scores),
            },
        )
        return (image, torch.cat(masks, dim=0), pipeline_info(info, block), preview)


def _shape_info(shape: str, outline: bool, boxes_only: bool) -> str:
    # Said out loud: outlines asked for and not given quietly become boxes otherwise.
    if shape != DETECT_SHAPE_SEGMENT:
        return shape
    if boxes_only:
        return f"{shape} (boxes: no masks from the model and none from the segmenter)"
    return f"{shape} (segmenter)" if outline else shape


def _draw_boxes(frame, found: list[tuple[Box, float, int | None]]) -> torch.Tensor:
    # With nothing to draw the preview is the frame itself, and the PIL round trip is 28ms of pure copying
    # per frame at 1920x1280 - nothing on one image, seconds on a video batch.
    if not found:
        return frame.unsqueeze(0)

    image = to_pil_rgb(frame)
    canvas = ImageDraw.Draw(image)

    # Scaled to the frame: PIL's default font is a ~10px bitmap, which is a speck on a 1024x1536 image.
    scale = max(image.width, image.height) / DETECT_PREVIEW_REFERENCE_SIZE
    width = max(1, round(DETECT_PREVIEW_WIDTH * scale))
    font_size = max(DETECT_PREVIEW_FONT_MIN, round(DETECT_PREVIEW_FONT_SIZE * scale))
    font = ImageFont.load_default(size=font_size)

    for (left, top, right, bottom), score, number in found:
        canvas.rectangle((left, top, right - 1, bottom - 1), outline=DETECT_PREVIEW_COLOR, width=width)

        # The number is the one a `1: tags` line in the detailer prompt addresses. A box that did not make
        # it into a region - dropped by the region limits - is labelled with a dash instead.
        label = f"{number}: {score:.2f}" if number is not None else f"-: {score:.2f}"
        text_top = top - font_size - width
        # The label goes inside the box when there is no room for it above.
        canvas.text((left + width, text_top if text_top >= 0 else top + width), label, fill=DETECT_PREVIEW_COLOR, font=font)

    return pil2tensor(image)


def _score_summary(scores: list[float]) -> str | None:
    if not scores:
        return None

    shown = ", ".join(f"{score:.2f}" for score in scores[:DETECT_INFO_SCORES])
    if len(scores) <= DETECT_INFO_SCORES:
        return shown

    return f"{shown} (+{len(scores) - DETECT_INFO_SCORES} more)"
