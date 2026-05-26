"""Face detection using SCRFD (Sample and Computation Redistribution for Face Detection).

SCRFD-10GF from InsightFace antelopev2 pack.
- Input: RGB image (any resolution).
- Output: list of FaceBox (x, y, w, h in pixel coords + 5-point landmarks).
- Handles small faces down to ~4px in 640px image (vs MediaPipe's ~20px floor).
- Resizes input to 640×640 before inference, scales boxes back to original coords.
- Reference: github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/scrfd.py
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
import structlog

from app.types import FaceBox

_SCRFD_MODEL_FILENAME = "scrfd_10g_bnkps.onnx"

# SCRFD-10GF configuration (9 output tensors: 3 strides × {score, bbox, kps}).
_INPUT_SIZE = (640, 640)      # (W, H) — standard SCRFD inference resolution
_STRIDES = [8, 16, 32]       # FPN feature map strides
_FMC = 3                     # number of feature map channels
_NUM_ANCHORS = 2             # anchors per cell per stride
_NMS_THRESHOLD = 0.4         # IoU threshold for NMS deduplication
_SCORE_THRESHOLD = 0.5       # Detection confidence — SCRFD is confident; do NOT lower

# Preprocessing constants (same as ArcFace).
_INPUT_MEAN = 127.5
_INPUT_STD = 128.0


def _get_model_path() -> Path:
    """Return path to SCRFD ONNX model. Must be baked into Docker image."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "models" / _SCRFD_MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"SCRFD model not found at {path}. "
            "Download scrfd_10g_bnkps.onnx from InsightFace antelopev2 pack "
            "and place in face-service/models/ before building."
        )
    return path


_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    """Process-level singleton for SCRFD ONNX inference session."""
    global _session
    if _session is None:
        model_path = _get_model_path()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _session


# ---------------------------------------------------------------------------
# Anchor generation (computed once at module load)
# ---------------------------------------------------------------------------

def _generate_anchor_centers(height: int, width: int, stride: int) -> np.ndarray:
    """Generate anchor center points for a single FPN stride level.

    Returns (height * width * _NUM_ANCHORS, 2) float32 array of (x, y) centers.
    """
    # np.mgrid returns [row_indices, col_indices] — we reverse to get [x, y]
    anchor_centers = np.stack(
        np.mgrid[:height, :width][::-1], axis=-1
    ).astype(np.float32)
    anchor_centers = (anchor_centers * stride).reshape(-1, 2)
    if _NUM_ANCHORS > 1:
        anchor_centers = np.stack(
            [anchor_centers] * _NUM_ANCHORS, axis=1
        ).reshape(-1, 2)
    return anchor_centers


# Pre-computed for 640×640 input. Cache keyed by (height, width, stride).
_anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}


def _get_anchors(height: int, width: int, stride: int) -> np.ndarray:
    key = (height, width, stride)
    if key not in _anchor_cache:
        _anchor_cache[key] = _generate_anchor_centers(height, width, stride)
    return _anchor_cache[key]


# ---------------------------------------------------------------------------
# Coordinate decoding (matches InsightFace distance2bbox / distance2kps)
# ---------------------------------------------------------------------------

def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode anchor-relative distance predictions to [x1, y1, x2, y2] bboxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode anchor-relative distance predictions to (N, 5, 2) landmark array."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1).reshape(-1, 5, 2)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    """Standard NMS on (N, 5) array of [x1, y1, x2, y2, score]."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _forward(
    session: ort.InferenceSession,
    det_img: np.ndarray,
    det_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SCRFD inference on a preprocessed 640×640 image.

    Args:
        session:   ONNX inference session.
        det_img:   Preprocessed image (H, W, 3) uint8, padded to _INPUT_SIZE.
        det_scale: Scale factor from original → detection image.

    Returns:
        pre_det: (N, 5) float32 array of [x1, y1, x2, y2, score] in original coords.
        kpss:    (N, 5, 2) float32 array of landmark coords in original coords.
    """
    # Build NCHW blob: subtract mean, divide by std, swap BGR→RGB (our input is RGB
    # but blobFromImage expects BGR, so we pass swapRB=True to convert RGB→BGR→RGB — no-op.
    # Actually: our image is already RGB. cv2.dnn.blobFromImage with swapRB=True would
    # swap it to BGR. We need to match InsightFace's convention which calls blobFromImage
    # on BGR images with swapRB=True. Since our input is RGB, we should pass swapRB=False
    # to keep it as-is — BUT the model was trained with swapRB=True on BGR input, which
    # produces RGB internally. So our RGB input should NOT be swapped.
    blob = cv2.dnn.blobFromImage(
        det_img,
        1.0 / _INPUT_STD,
        _INPUT_SIZE,
        (_INPUT_MEAN, _INPUT_MEAN, _INPUT_MEAN),
        swapRB=False,  # input is already RGB
    )

    input_name = session.get_inputs()[0].name
    net_outs = session.run(None, {input_name: blob})

    input_height, input_width = _INPUT_SIZE[1], _INPUT_SIZE[0]
    scores_list: list[np.ndarray] = []
    bboxes_list: list[np.ndarray] = []
    kpss_list: list[np.ndarray] = []

    for idx, stride in enumerate(_STRIDES):
        scores = net_outs[idx]           # (N_anchors, 1)
        bbox_preds = net_outs[idx + _FMC] * stride  # (N_anchors, 4)
        kps_preds = net_outs[idx + _FMC * 2] * stride  # (N_anchors, 10)

        height = input_height // stride
        width = input_width // stride
        anchor_centers = _get_anchors(height, width, stride)

        pos_inds = np.where(scores >= _SCORE_THRESHOLD)[0]
        if pos_inds.size == 0:
            continue

        pos_scores = scores[pos_inds]
        bboxes = _distance2bbox(anchor_centers, bbox_preds)
        pos_bboxes = bboxes[pos_inds]
        kpss = _distance2kps(anchor_centers, kps_preds)
        pos_kpss = kpss[pos_inds]

        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)
        kpss_list.append(pos_kpss)

    if not scores_list:
        return (
            np.empty((0, 5), dtype=np.float32),
            np.empty((0, 5, 2), dtype=np.float32),
        )

    scores_all = np.vstack(scores_list)
    scores_ravel = scores_all.ravel()
    order = scores_ravel.argsort()[::-1]
    bboxes_all = np.vstack(bboxes_list) / det_scale
    kpss_all = np.vstack(kpss_list) / det_scale

    pre_det = np.hstack((bboxes_all, scores_all)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    kpss_all = kpss_all[order, :, :]

    return pre_det, kpss_all


# ---------------------------------------------------------------------------
# Public API (drop-in replacement for detector_mediapipe)
# ---------------------------------------------------------------------------

def detect_faces(
    rgb_image: np.ndarray,
    model_selection: int = 1,  # kept for call-site compatibility, unused
    max_faces: int = 20,
    **kwargs,  # swallow min_detection_confidence / _use_low_conf for legacy scripts compatibility
) -> List[FaceBox]:
    """Detect faces using SCRFD-10GF. Drop-in replacement for detector_mediapipe.detect_faces().

    Args:
        rgb_image:       Full RGB image (H, W, 3) uint8.
        model_selection: Ignored — kept for API compatibility with MediaPipe caller.
        max_faces:       Maximum faces to return, ordered by confidence descending.

    Returns:
        List of FaceBox in pixel coords. Empty if no faces detected.
    """
    log = structlog.get_logger()

    if "min_detection_confidence" in kwargs:
        log.debug(
            "detect_faces: min_detection_confidence is ignored by SCRFD — "
            "threshold is fixed at _SCORE_THRESHOLD=0.5"
        )

    if rgb_image is None or rgb_image.size == 0:
        return []

    session = _get_session()
    img_h, img_w = rgb_image.shape[:2]

    # Resize to fit _INPUT_SIZE while maintaining aspect ratio.
    # Padding is top-left aligned (InsightFace convention).
    target_w, target_h = _INPUT_SIZE
    im_ratio = float(img_h) / img_w
    model_ratio = float(target_h) / target_w
    if im_ratio > model_ratio:
        new_height = target_h
        new_width = int(new_height / im_ratio)
    else:
        new_width = target_w
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img_h

    resized = cv2.resize(rgb_image, (new_width, new_height))
    det_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized

    pre_det, kpss = _forward(session, det_img, det_scale)

    if pre_det.shape[0] == 0:
        return []

    # NMS deduplication
    keep = _nms(pre_det, _NMS_THRESHOLD)
    det = pre_det[keep, :]
    kpss = kpss[keep, :, :]

    # Sort by score descending, limit to max_faces
    score_order = det[:, 4].argsort()[::-1][:max_faces]
    det = det[score_order, :]
    kpss = kpss[score_order, :, :]

    result: List[FaceBox] = []
    for i in range(det.shape[0]):
        x1, y1, x2, y2, score = det[i]
        # Clamp to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img_w, int(x2))
        y2 = min(img_h, int(y2))
        bw, bh = x2 - x1, y2 - y1

        if bw <= 0 or bh <= 0:
            continue

        # SCRFD 5-point landmark order: left_eye, right_eye, nose, mouth_left, mouth_right
        # — identical to ArcFace canonical order used by align_face_5pt().
        pts = kpss[i]
        eye_left = (int(pts[0, 0]), int(pts[0, 1]))
        eye_right = (int(pts[1, 0]), int(pts[1, 1]))
        nose = (int(pts[2, 0]), int(pts[2, 1]))
        mouth_left = (int(pts[3, 0]), int(pts[3, 1]))
        mouth_right = (int(pts[4, 0]), int(pts[4, 1]))

        result.append(FaceBox(
            x=x1, y=y1, w=bw, h=bh,
            eye_left=eye_left,
            eye_right=eye_right,
            nose=nose,
            mouth_left=mouth_left,
            mouth_right=mouth_right,
        ))

    log.info("scrfd-detect", num_faces=len(result), img_size=f"{img_w}x{img_h}")
    return result


def get_largest_face(boxes: List[FaceBox]) -> FaceBox | None:
    """Return the face box with the largest area, or None if empty."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: b.area)
