from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

from config import CFG, MODELS_DIR


MASK_CLASS_NAME = "with_mask"
NO_MASK_CLASS_NAME = "without_mask"


def configure_output_encoding() -> None:
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _download(url: str, dst: str) -> None:
    print(f"   Downloading {Path(dst).name} ...")

    def _progress(count, block, total):
        if total > 0:
            pct = min(count * block / total * 100, 100)
            print(f"\r      {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dst, reporthook=_progress)
    print()


def load_face_detector():
    prototxt = CFG.FACE_PROTOTXT
    weights = CFG.FACE_WEIGHTS
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(prototxt).exists():
        _download(
            "https://raw.githubusercontent.com/opencv/opencv/master/"
            "samples/dnn/face_detector/deploy.prototxt",
            prototxt,
        )
    if not Path(weights).exists():
        _download(
            "https://github.com/opencv/opencv_3rdparty/raw/"
            "dnn_samples_face_detector_20170830/"
            "res10_300x300_ssd_iter_140000.caffemodel",
            weights,
        )

    return cv2.dnn.readNet(prototxt, weights)


def detect_faces(frame: np.ndarray, net, confidence_threshold: float | None = None) -> list:
    threshold = CFG.FACE_CONFIDENCE if confidence_threshold is None else confidence_threshold

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for idx in range(detections.shape[2]):
        confidence = float(detections[0, 0, idx, 2])
        if confidence < threshold:
            continue

        box = detections[0, 0, idx, 3:7] * np.array([width, height, width, height])
        x1, y1, x2, y2 = box.astype("int")
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        if x2 > x1 and y2 > y1:
            faces.append((x1, y1, x2, y2, confidence))
    return faces


def expand_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    padding_ratio: float | None = None,
) -> tuple[int, int, int, int]:
    ratio = CFG.FACE_PADDING_RATIO if padding_ratio is None else padding_ratio
    pad_x = int((x2 - x1) * ratio)
    pad_y = int((y2 - y1) * ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width - 1, x2 + pad_x)
    y2 = min(height - 1, y2 + pad_y)
    return x1, y1, x2, y2


def crop_face(frame: np.ndarray, face_box: tuple, padding_ratio: float | None = None):
    height, width = frame.shape[:2]
    x1, y1, x2, y2, _ = face_box
    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, width, height, padding_ratio)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def prepare_classifier_input(image_rgb: np.ndarray) -> np.ndarray:
    image = cv2.resize(image_rgb, CFG.IMAGE_SIZE)
    return img_to_array(image).astype("float32") / 255.0


def _class_indices() -> tuple[int, int]:
    mapping = {name: idx for idx, name in enumerate(CFG.CLASSES)}
    if MASK_CLASS_NAME not in mapping or NO_MASK_CLASS_NAME not in mapping:
        raise ValueError(
            f"CFG.CLASSES must contain '{MASK_CLASS_NAME}' and '{NO_MASK_CLASS_NAME}'."
        )
    return mapping[MASK_CLASS_NAME], mapping[NO_MASK_CLASS_NAME]


def decode_prediction(probs: np.ndarray) -> tuple[str, float, bool]:
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    mask_idx, _ = _class_indices()
    pred_idx = int(np.argmax(probs))
    is_mask = pred_idx == mask_idx
    confidence = float(probs[pred_idx])
    label = "Mask" if is_mask else "No Mask"
    return label, confidence, is_mask


@dataclass
class _TrackState:
    center: tuple[float, float]
    probs: np.ndarray
    missed: int = 0


class TemporalSmoother:
    def __init__(self, alpha: float, max_distance: int, ttl: int):
        self.alpha = float(alpha)
        self.max_distance = int(max_distance)
        self.ttl = int(ttl)
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 0

    def update(self, faces: list, probs_list: list[np.ndarray]) -> list[np.ndarray]:
        smoothed = []
        used_track_ids = set()

        for face, probs in zip(faces, probs_list):
            x1, y1, x2, y2, _ = face
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            probs = np.asarray(probs, dtype=np.float32)

            best_track_id = None
            best_distance = float("inf")
            for track_id, state in self._tracks.items():
                if track_id in used_track_ids:
                    continue
                distance = float(np.hypot(center[0] - state.center[0], center[1] - state.center[1]))
                if distance < best_distance and distance <= self.max_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _TrackState(center=center, probs=probs.copy())
                used_track_ids.add(track_id)
                smoothed.append(probs)
                continue

            state = self._tracks[best_track_id]
            state.probs = self.alpha * state.probs + (1.0 - self.alpha) * probs
            state.center = center
            state.missed = 0
            used_track_ids.add(best_track_id)
            smoothed.append(state.probs.copy())

        for track_id in list(self._tracks.keys()):
            if track_id in used_track_ids:
                continue
            self._tracks[track_id].missed += 1
            if self._tracks[track_id].missed > self.ttl:
                del self._tracks[track_id]

        return smoothed
