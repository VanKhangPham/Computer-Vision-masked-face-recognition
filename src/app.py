
import os
import sys
import time
import argparse
import urllib.request
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Import cấu hình trung tâm ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CFG, MODELS_DIR, SCREENSHOTS


# ══════════════════════════════════════════════════════════
#  MÀU SẮC & KIỂU HIỂN THỊ (BGR)
# ══════════════════════════════════════════════════════════
GREEN  = (50, 205, 50)
RED    = (50,  50, 220)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
YELLOW = (0,  200, 200)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2


# ══════════════════════════════════════════════════════════
#  BƯỚC 1 – LOAD MÔ HÌNH & FACE DETECTOR
# ══════════════════════════════════════════════════════════
def _download(url: str, dst: str) -> None:
    """Tải file về nếu chưa có, hiển thị progress."""
    print(f"   📥 Đang tải {Path(dst).name} ...")
    def _progress(count, block, total):
        if total > 0:
            pct = min(count * block / total * 100, 100)
            print(f"\r      {pct:5.1f}%", end="", flush=True)
    urllib.request.urlretrieve(url, dst, reporthook=_progress)
    print()   # newline sau progress bar


def load_face_detector():
    """
    Load face detector OpenCV DNN (Caffe SSD – chính xác hơn Haar).
    Tự tải file nếu chưa có (~10 MB).
    """
    prototxt = CFG.FACE_PROTOTXT
    weights  = CFG.FACE_WEIGHTS
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

    net = cv2.dnn.readNet(prototxt, weights)
    print("  ✅ Face detector sẵn sàng (OpenCV DNN – Caffe SSD)")
    return net


def load_mask_model():
    """Load model phân loại mask. Báo lỗi rõ ràng nếu chưa train."""
    path = CFG.MASK_MODEL
    if not Path(path).exists():
        print(f"\n  ❌ Không tìm thấy model: {path}")
        print("  → Hãy chạy train_model.py (Người 2) trước!\n")
        sys.exit(1)
    model = keras_load(path)
    print(f"  ✅ Mask model sẵn sàng: {Path(path).name}")
    return model


# ══════════════════════════════════════════════════════════
#  BƯỚC 2 – PHÁT HIỆN KHUÔN MẶT
# ══════════════════════════════════════════════════════════
def detect_faces(frame: np.ndarray, net) -> list:
    """
    Trả về list [(x1, y1, x2, y2, confidence), ...].
    Dùng OpenCV DNN để hỗ trợ nhiều góc độ & ánh sáng kém.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    dets = net.forward()

    faces = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < CFG.FACE_CONFIDENCE:
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")
        # Clamp toạ độ vào kích thước frame
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        if x2 > x1 and y2 > y1:           # Loại box suy biến
            faces.append((x1, y1, x2, y2, conf))
    return faces


# ══════════════════════════════════════════════════════════
#  BƯỚC 3 – PHÂN LOẠI MASK
# ══════════════════════════════════════════════════════════
@dataclass
class Prediction:
    label     : str    # "Mask" hoặc "No Mask"
    confidence: float
    is_mask   : bool

    @property
    def color(self):
        return GREEN if self.is_mask else RED

    @property
    def text(self):
        return f"{self.label}: {self.confidence:.0%}"


def predict_masks(frame: np.ndarray, faces: list, model) -> list[Prediction]:
    """Phân loại mask cho từng khuôn mặt."""
    if not faces:
        return []

    crops = []
    for (x1, y1, x2, y2, _) in faces:
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            crops.append(None)
            continue
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, CFG.IMAGE_SIZE)
        face = preprocess_input(img_to_array(face))
        crops.append(face)

    # Lọc và batch predict
    valid_crops  = [c for c in crops if c is not None]
    if not valid_crops:
        return []

    probs = model.predict(np.array(valid_crops), verbose=0)

    results = []
    prob_iter = iter(probs)
    for crop in crops:
        if crop is None:
            results.append(Prediction("No Mask", 0.0, False))
            continue
        p = next(prob_iter)
        is_mask = float(p[0]) >= CFG.MASK_THRESHOLD
        results.append(Prediction(
            label      = "Mask" if is_mask else "No Mask",
            confidence = float(p[0]) if is_mask else float(p[1]),
            is_mask    = is_mask,
        ))
    return results


# ══════════════════════════════════════════════════════════
#  BƯỚC 4 – VẼ KẾT QUẢ LÊN FRAME
# ══════════════════════════════════════════════════════════
def _draw_corner(frame, x1, y1, x2, y2, color, length=18, thick=3):
    """Vẽ 4 góc viền thay thế rectangle đầy đủ → trông hiện đại hơn."""
    for (px, py, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * length, py), color, thick)
        cv2.line(frame, (px, py), (px, py + dy * length), color, thick)


def draw_results(frame: np.ndarray, faces: list,
                 preds: list[Prediction]) -> np.ndarray:
    """Vẽ bounding box, label, corner accent lên frame."""
    for (x1, y1, x2, y2, _), pred in zip(faces, preds):
        # Viền chính (mỏng)
        cv2.rectangle(frame, (x1, y1), (x2, y2), pred.color, 1)
        # Góc accent (dày)
        _draw_corner(frame, x1, y1, x2, y2, pred.color)

        # Label background
        (tw, th), _ = cv2.getTextSize(pred.text, FONT, FONT_SCALE, THICKNESS)
        label_y = y1 - 10 if y1 - 10 > th + 8 else y2 + th + 10
        cv2.rectangle(frame,
                      (x1, label_y - th - 8), (x1 + tw + 10, label_y + 4),
                      pred.color, cv2.FILLED)
        cv2.putText(frame, pred.text, (x1 + 5, label_y - 2),
                    FONT, FONT_SCALE, WHITE, THICKNESS)
    return frame


def draw_hud(frame: np.ndarray, preds: list[Prediction],
             fps: float) -> np.ndarray:
    """HUD ở góc trên trái: FPS, số khuôn mặt, thống kê."""
    mask_n    = sum(1 for p in preds if p.is_mask)
    no_mask_n = len(preds) - mask_n

    # Nền bán trong suốt
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (210, 100), BLACK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    lines = [
        (f"FPS: {fps:5.1f}",                (200, 200, 200)),
        (f"Mat: {len(preds)}",              (200, 200, 200)),
        (f"Mask: {mask_n}",                 GREEN),
        (f"No Mask: {no_mask_n}",           RED),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (8, 22 + i * 22),
                    FONT, 0.55, color, 1)
    return frame


# ══════════════════════════════════════════════════════════
#  CHẾ ĐỘ 1 – WEBCAM REALTIME
# ══════════════════════════════════════════════════════════
def run_webcam(face_net, mask_model) -> None:
    print("\n  📹 WEBCAM MODE")
    print("  ─────────────────────────────────────")
    print("  [Q]  Thoát")
    print("  [S]  Chụp ảnh màn hình")
    print("  [P]  Tạm dừng / Tiếp tục")
    print("  ─────────────────────────────────────\n")

    cap = cv2.VideoCapture(CFG.CAMERA_ID)
    if not cap.isOpened():
        print(f"  ❌ Không mở được camera {CFG.CAMERA_ID}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.CAMERA_HEIGHT)

    paused    = False
    prev_time = time.time()
    shot_idx  = 0
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("  ❌ Không đọc được frame")
                break

            faces = detect_faces(frame, face_net)
            preds = predict_masks(frame, faces, mask_model)

            curr_time = time.time()
            fps       = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            frame = draw_results(frame, faces, preds)
            frame = draw_hud(frame, preds, fps)

        cv2.imshow("Face Mask Detection  |  Q=thoat  S=chup  P=tam_dung", frame)
        key = cv2.waitKey(1) & 0xFF

        if   key == ord("q"):  break
        elif key == ord("s"):
            path = SCREENSHOTS / f"shot_{shot_idx:04d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"  📸 Đã lưu: {path}")
            shot_idx += 1
        elif key == ord("p"):
            paused = not paused
            status = "⏸ TẠM DỪNG" if paused else "▶ TIẾP TỤC"
            print(f"  {status}")

    cap.release()
    cv2.destroyAllWindows()
    print("  ✅ Đã thoát webcam.")


# ══════════════════════════════════════════════════════════
#  CHẾ ĐỘ 2 – ẢNH ĐƠN LẺ
# ══════════════════════════════════════════════════════════
def run_image(path: str, face_net, mask_model,
              show: bool = True, save: bool = True) -> dict:
    """Nhận diện mask trong 1 ảnh tĩnh."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"  ❌ Không đọc được ảnh: {path}")
        return {}

    t0    = time.perf_counter()
    faces = detect_faces(frame, face_net)
    preds = predict_masks(frame, faces, mask_model)
    ms    = (time.perf_counter() - t0) * 1000

    frame = draw_results(frame, faces, preds)

    # Watermark thời gian
    cv2.putText(frame, f"{ms:.0f} ms",
                (frame.shape[1] - 80, frame.shape[0] - 10),
                FONT, 0.45, YELLOW, 1)

    # In kết quả
    print(f"\n  📷  {Path(path).name}")
    print(f"  Thời gian  : {ms:.0f} ms")
    print(f"  Khuôn mặt : {len(faces)}")
    for i, ((x1, y1, x2, y2, fc), pred) in enumerate(zip(faces, preds), 1):
        print(f"    [{i}] {pred.label:8s} {pred.confidence:.0%}"
              f"  | face_conf={fc:.0%}  box=({x1},{y1})→({x2},{y2})")

    if save:
        out = Path("results") / f"pred_{Path(path).name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame)
        print(f"  💾 Lưu kết quả: {out}")

    if show:
        cv2.imshow(f"[Q] đóng – {Path(path).name}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "file"       : path,
        "faces"      : len(faces),
        "predictions": [(p.label, p.confidence) for p in preds],
        "elapsed_ms" : ms,
    }


# ══════════════════════════════════════════════════════════
#  CHẾ ĐỘ 3 – BATCH (THƯ MỤC)
# ══════════════════════════════════════════════════════════
def run_batch(folder: str, face_net, mask_model) -> None:
    """Xử lý tất cả ảnh trong 1 thư mục."""
    EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in Path(folder).rglob("*")
              if p.suffix.lower() in EXTS]

    if not images:
        print(f"  ❌ Không tìm thấy ảnh trong: {folder}")
        return

    print(f"\n  📂 BATCH MODE – {len(images)} ảnh")
    print("  ─" * 25)

    all_results  = []
    total_mask   = 0
    total_nomask = 0

    for i, img_path in enumerate(images, 1):
        print(f"  [{i:3d}/{len(images)}]", end=" ")
        r = run_image(str(img_path), face_net, mask_model,
                      show=False, save=True)
        if r:
            all_results.append(r)
            for label, _ in r.get("predictions", []):
                if label == "Mask": total_mask   += 1
                else:               total_nomask += 1

    # Tổng kết
    avg_ms = np.mean([r["elapsed_ms"] for r in all_results]) if all_results else 0
    print("\n" + "═" * 45)
    print(f"  ✅ Hoàn tất batch processing")
    print(f"  Số ảnh đã xử lý   : {len(all_results)}")
    print(f"  Tổng khuôn mặt    : {total_mask + total_nomask}")
    print(f"  ✅ Đeo khẩu trang  : {total_mask}")
    print(f"  ❌ Không đeo        : {total_nomask}")
    print(f"  Thời gian TB/ảnh  : {avg_ms:.0f} ms")
    print("═" * 45)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Face Mask Detection – Demo App",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["webcam", "image", "batch"], default="webcam",
        help="webcam  – realtime từ webcam\n"
             "image   – xử lý 1 ảnh\n"
             "batch   – xử lý cả thư mục",
    )
    parser.add_argument("--input",   default=None, help="Đường dẫn ảnh hoặc thư mục")
    parser.add_argument("--no-show", action="store_true",
                        help="Không hiển thị cửa sổ (headless / Colab)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║  FACE MASK – BƯỚC 3: DEMO APPLICATION   ║")
    print("╚══════════════════════════════════════════╝\n")

    # Load models
    face_net   = load_face_detector()
    mask_model = load_mask_model()

    # Phân nhánh
    if args.mode == "webcam":
        run_webcam(face_net, mask_model)

    elif args.mode == "image":
        if not args.input:
            parser.error("--mode image cần --input <đường dẫn ảnh>")
        run_image(args.input, face_net, mask_model,
                  show=not args.no_show, save=True)

    elif args.mode == "batch":
        if not args.input:
            parser.error("--mode batch cần --input <thư mục>")
        run_batch(args.input, face_net, mask_model)


if __name__ == "__main__":
    main()