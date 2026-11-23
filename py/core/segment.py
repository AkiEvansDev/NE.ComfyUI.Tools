"""Box-prompted segmentation: an outline for every detection a box-only model hands over.

There is no hand segmentation model among the YOLO exports, so `segment` on a hand detector would otherwise
fall back to boxes. MobileSAM (encoder and single-mask decoder as ONNX, from `Acly/MobileSAM`, MIT) turns
each box into the outline of what it holds. It runs through the same onnxruntime sessions as the detector.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from ...constants import (
    SEGMENT_CONTEXT_FACTOR,
    SEGMENT_DECODER_FILE,
    SEGMENT_ENCODER_FILE,
    SEGMENT_INPUT_SIZE,
    SEGMENT_MODELS_RELATIVE_PATH,
)
from .detect import Box, Detection, load_session


def segmenter_available() -> bool:
    return all(os.path.isfile(path) for path in _model_paths())


def _model_paths() -> tuple[str, str]:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    folder = os.path.join(root, *SEGMENT_MODELS_RELATIVE_PATH)
    return os.path.join(folder, SEGMENT_ENCODER_FILE), os.path.join(folder, SEGMENT_DECODER_FILE)


def outline_detections(image: torch.Tensor, detections: list[Detection]) -> list[Detection]:
    """Every detection without a mask of its own gets the outline of what its box holds."""
    if all(detection.mask is not None for detection in detections):
        return detections

    frame = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return [detection if detection.mask is not None else _outlined(frame, detection) for detection in detections]


def _outlined(frame: np.ndarray, detection: Detection) -> Detection:
    mask = segment_box(frame, detection.box)
    # An empty answer keeps the box: a hand the segmenter cannot see is still a hand the detector found.
    return Detection(detection.box, detection.score, mask if mask.any() else None)


def segment_box(frame: np.ndarray, box: Box) -> np.ndarray:
    """The (H, W) bool outline of the object inside `box`, cut to the box.

    The encoder sees a crop around the box rather than the frame: at 1024 on the frame's long side a hand
    covers a few cells of the 64x64 embedding and the fingers come back as a blob. With SEGMENT_CONTEXT_FACTOR
    around it the hand fills half the encoder's input and the outline follows the fingers.
    """
    encoder_path, decoder_path = _model_paths()
    height, width = frame.shape[:2]
    left, top, right, bottom = box

    crop_left, crop_top, crop_right, crop_bottom = _context(box, width, height)
    patch = frame[crop_top:crop_bottom, crop_left:crop_right]
    patch_height, patch_width = patch.shape[:2]

    # The encoder normalises and pads to a square itself; it wants RGB 0..255 with the long side at exactly 1024.
    scale = SEGMENT_INPUT_SIZE / max(patch_width, patch_height)
    size = (int(patch_width * scale + 0.5), int(patch_height * scale + 0.5))
    resized = cv2.resize(patch, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    embeddings = load_session(encoder_path).run(None, {"input_image": resized.astype(np.float32)})[0]

    # The box as SAM's two corner points (labels 2 and 3), plus a padding point: without it the single-mask
    # decoder skips the box mask and returns one of its ambiguous alternatives.
    points = np.array([[[left - crop_left, top - crop_top], [right - crop_left, bottom - crop_top], [0, 0]]], dtype=np.float32)
    points[:, :2] *= scale
    masks = load_session(decoder_path).run(None, {
        "image_embeddings": embeddings,
        "point_coords": points,
        "point_labels": np.array([[2, 3, -1]], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([patch_height, patch_width], dtype=np.float32),
    })[0]

    # Logits at the crop's own size; above zero is inside.
    solid = masks[0, 0] > 0.0
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = solid[top - crop_top:bottom - crop_top, left - crop_left:right - crop_left]
    return mask


def _context(box: Box, width: int, height: int) -> Box:
    left, top, right, bottom = box
    grow_x = int(round((right - left) * (SEGMENT_CONTEXT_FACTOR - 1.0) / 2.0))
    grow_y = int(round((bottom - top) * (SEGMENT_CONTEXT_FACTOR - 1.0) / 2.0))
    return (max(0, left - grow_x), max(0, top - grow_y), min(width, right + grow_x), min(height, bottom + grow_y))
