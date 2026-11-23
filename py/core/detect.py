"""YOLO detection through onnxruntime.

A YOLO model exported to ONNX runs on onnxruntime's CUDA provider, keeping the forward pass on the GPU
where the rest of the graph already lives. It used to run on OpenCV's DNN module, which has no CUDA
build on PyPI: that forward was the only multi-core CPU work in the whole pipeline, ~150ms with every
core in the ceiling, against ~4ms here. cv2 is still used for the image work and for NMS.
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime
# Also load-bearing for the CUDA provider: cudnn64_9.dll ships inside torch/lib, and onnxruntime only
# finds it once torch has been imported. Without it a session still reports CUDA and then fails the run.
import torch

from ...constants import (
    DETECT_CUDA_PROVIDER,
    DETECT_ELLIPSE_CORNER,
    DETECT_EXECUTION_PROVIDERS,
    DETECT_INPUT_SIZE,
    DETECT_LOG_SEVERITY,
    DETECT_MODEL_EXTENSION,
    DETECT_MODELS_RELATIVE_PATH,
    DETECT_PADDING_COLOR,
    DETECT_SHAPE_ELLIPSE,
    DETECT_SHAPE_SEGMENT,
    DETECT_SPINNING_OPTIONS,
)

Box = tuple[int, int, int, int]


class Detection(NamedTuple):
    box: Box
    score: float
    # The object's own pixels, (H, W) bool in frame coordinates - None from a model that only finds boxes.
    mask: np.ndarray | None = None

_sessions: dict[str, onnxruntime.InferenceSession] = {}


def detect_model_names() -> list[str]:
    folder = _detect_folder()
    if not os.path.isdir(folder):
        return []

    return sorted(name for name in os.listdir(folder) if name.lower().endswith(DETECT_MODEL_EXTENSION))


def detect_model_path(name: str) -> str:
    return os.path.join(_detect_folder(), name)


def _detect_folder() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(root, *DETECT_MODELS_RELATIVE_PATH)


def load_session(path: str) -> onnxruntime.InferenceSession:
    """An onnxruntime session for the detector or the segmenter, cached by path.

    Re-reading the ONNX on every prompt costs a second for nothing.
    """
    if path not in _sessions:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found: {path}")

        session = onnxruntime.InferenceSession(path, _session_options(), providers=list(DETECT_EXECUTION_PROVIDERS))
        # Said out loud on purpose: falling back to the CPU provider is the multi-core spike coming back.
        print(f"[comfyui-ne-tools] {os.path.basename(path)} runs on {session.get_providers()[0]}.")
        _sessions[path] = session

    return _sessions[path]


def _session_options() -> onnxruntime.SessionOptions:
    """Keeps the detector off the CPU, which it otherwise claims for far longer than the forward takes."""
    options = onnxruntime.SessionOptions()
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    # Errors only: the SAM encoder's graph gets memcpy nodes on CUDA, and onnxruntime warns about it on every load.
    options.log_severity_level = DETECT_LOG_SEVERITY

    # onnxruntime's thread pools busy-wait after every run by default. The forward itself is ~4ms on the GPU
    # and then every core spins for a few hundred more, which is the whole of the spike this node produced.
    for entry in DETECT_SPINNING_OPTIONS:
        options.add_session_config_entry(entry, "0")

    # With the forward on CUDA the pools only shuffle inputs, so one thread each is plenty. On the fallback
    # path they run the model instead, and starving them there would cost far more than the spinning did.
    if DETECT_CUDA_PROVIDER in onnxruntime.get_available_providers():
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

    return options


def detect_objects(image: torch.Tensor, session: onnxruntime.InferenceSession, confidence: float, iou: float) -> list[Detection]:
    """Detections in the coordinates of the given (H, W, 3) frame, best score first."""
    frame = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    height, width = frame.shape[:2]

    blob, scale, pad_x, pad_y = _letterbox_blob(frame)
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    # A segmentation export carries the mask prototypes as a second output, (1, masks, 160, 160).
    protos = outputs[1][0] if len(outputs) > 1 else None

    return _decode(outputs[0], protos, confidence, iou, scale, pad_x, pad_y, width, height)


def _letterbox_blob(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Aspect-preserving fit into a square, the way the model was trained.

    Stretching a 832x1216 frame into a square instead would squash every hand in it.
    """
    height, width = frame.shape[:2]
    scale = DETECT_INPUT_SIZE / max(width, height)
    scaled_width = int(round(width * scale))
    scaled_height = int(round(height * scale))

    canvas = np.full((DETECT_INPUT_SIZE, DETECT_INPUT_SIZE, 3), DETECT_PADDING_COLOR, dtype=np.uint8)
    pad_x = (DETECT_INPUT_SIZE - scaled_width) // 2
    pad_y = (DETECT_INPUT_SIZE - scaled_height) // 2
    canvas[pad_y:pad_y + scaled_height, pad_x:pad_x + scaled_width] = cv2.resize(
        frame,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # swapRB stays off: an IMAGE tensor is already RGB, which is what the model was fed during training.
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (DETECT_INPUT_SIZE, DETECT_INPUT_SIZE), swapRB=False, crop=False)
    return blob, scale, pad_x, pad_y


def _decode(
    output: np.ndarray,
    protos: np.ndarray | None,
    confidence: float,
    iou: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
) -> list[Detection]:
    # (1, 4 + classes [+ masks], anchors) -> (anchors, ...); the four are cx, cy, w, h on the letterboxed square.
    predictions = output[0].T
    # A segmentation export appends one coefficient per prototype after the class scores. Taking the maximum
    # over the whole tail read those coefficients as scores, and they are unbounded - every anchor passed.
    classes = predictions.shape[1] - 4 - (protos.shape[0] if protos is not None else 0)
    scores = predictions[:, 4:4 + classes].max(axis=1)
    kept = scores >= confidence
    predictions, scores = predictions[kept], scores[kept]

    if not len(predictions):
        return []

    boxes = []
    for center_x, center_y, box_width, box_height in predictions[:, :4]:
        boxes.append([
            (center_x - box_width / 2 - pad_x) / scale,
            (center_y - box_height / 2 - pad_y) / scale,
            box_width / scale,
            box_height / scale,
        ])

    indices = cv2.dnn.NMSBoxes(boxes, scores.tolist(), confidence, iou)
    if len(indices) == 0:
        return []

    found = []
    for index in np.array(indices).flatten():
        left, top, box_width, box_height = boxes[int(index)]
        box = (
            max(0, int(round(left))),
            max(0, int(round(top))),
            min(width, int(round(left + box_width))),
            min(height, int(round(top + box_height))),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        mask = None
        if protos is not None:
            mask = _instance_mask(predictions[int(index), 4 + classes:], protos, box, scale, pad_x, pad_y, width, height)

        found.append(Detection(box, float(scores[int(index)]), mask))

    found.sort(key=lambda detection: detection.score, reverse=True)
    return found


def _instance_mask(
    coefficients: np.ndarray,
    protos: np.ndarray,
    box: Box,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """One object's mask: its coefficients over the prototypes, taken back out of the letterbox, cut to its box."""
    count, proto_height, proto_width = protos.shape
    logits = (coefficients @ protos.reshape(count, -1)).reshape(proto_height, proto_width)
    square = cv2.resize(logits, (DETECT_INPUT_SIZE, DETECT_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    inner = square[pad_y:pad_y + int(round(height * scale)), pad_x:pad_x + int(round(width * scale))]
    # A logit above zero is a sigmoid above one half, without computing the sigmoid over the whole frame.
    solid = cv2.resize(inner, (width, height), interpolation=cv2.INTER_LINEAR) > 0.0

    # The prototypes cover the whole image, so a mask spills onto every similar object; the box is what
    # says which one this is - the same crop ultralytics applies.
    left, top, right, bottom = box
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = solid[top:bottom, left:right]
    return mask


def grow_box(box: Box, dilation: int, width: int, height: int) -> Box | None:
    """Widen a box by the dilation, clamped to the frame. None when it collapses to nothing."""
    left, top, right, bottom = box
    left = max(0, left - dilation)
    top = max(0, top - dilation)
    right = min(width, right + dilation)
    bottom = min(height, bottom + dilation)

    if right <= left or bottom <= top:
        return None

    return (left, top, right, bottom)


def grow_detection(detection: Detection, percent: int, width: int, height: int) -> Detection | None:
    """The box and its outline grown by `percent` of the box's short side. None when either vanishes."""
    left, top, right, bottom = detection.box
    dilation = int(round(min(right - left, bottom - top) * percent / 100.0))
    box = grow_box(detection.box, dilation, width, height)
    if box is None:
        return None

    if detection.mask is None or dilation == 0:
        return Detection(box, detection.score, detection.mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(dilation) * 2 + 1, abs(dilation) * 2 + 1))
    operation = cv2.dilate if dilation > 0 else cv2.erode
    mask = operation(detection.mask.astype(np.uint8), kernel).astype(bool)

    if not mask.any():
        return None

    return Detection(box, detection.score, mask)


def detections_to_mask(detections: list[Detection], width: int, height: int, shape: str) -> torch.Tensor:
    """The MASK the detailer is handed. A detection without a mask of its own falls back to its box."""
    mask = torch.zeros((1, height, width), dtype=torch.float32)

    for detection in detections:
        left, top, right, bottom = detection.box

        if shape == DETECT_SHAPE_SEGMENT and detection.mask is not None:
            mask[0][torch.from_numpy(detection.mask)] = 1.0
        elif shape == DETECT_SHAPE_ELLIPSE:
            # The maximum, not an overwrite: an eased corner must not cut into a neighbour it overlaps.
            area = mask[0, top:bottom, left:right]
            torch.maximum(area, _ellipse(right - left, bottom - top), out=area)
        else:
            mask[0, top:bottom, left:right] = 1.0

    return mask


def _ellipse(width: int, height: int) -> torch.Tensor:
    """1.0 over the ellipse inscribed in the box, easing down to DETECT_ELLIPSE_CORNER in its corners.

    Solid all the way to the middle of every side, so the box and the blob the detailer finds keep their
    size; only the corners, which are almost always background, are handed to the sampler late.
    """
    rows = (torch.arange(height, dtype=torch.float32) + 0.5) / height * 2.0 - 1.0
    columns = (torch.arange(width, dtype=torch.float32) + 0.5) / width * 2.0 - 1.0
    distance = torch.sqrt(rows[:, None] ** 2 + columns[None, :] ** 2)
    ramp = ((distance - 1.0) / (math.sqrt(2.0) - 1.0)).clamp(0.0, 1.0)
    return 1.0 - (1.0 - DETECT_ELLIPSE_CORNER) * ramp * ramp * (3.0 - 2.0 * ramp)
