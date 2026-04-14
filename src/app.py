
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from tensorflow.keras.models import load_model as keras_load

# Import cấu hình trung tâm
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CFG, SCREENSHOTS
from src.vision_utils import (
    TemporalSmoother,
    configure_output_encoding,
    crop_face,
    decode_prediction,
    detect_faces,
    load_face_detector,
    prepare_classifier_input,
)


configure_output_encoding()

#  MÀU SẮC & KIỂU HIỂN THỊ (BGR)
GREEN  = (50, 205, 50)
RED    = (50,  50, 220)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
YELLOW = (0,  200, 200)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2


def load_mask_model():
    """Load model phân loại mask. Báo lỗi rõ ràng nếu chưa train."""
    path = CFG.MASK_MODEL
    if not Path(path).exists():
        print(f"\n   Không tìm thấy model: {path}")
        print("  →Hãy chạy train_model.py trước!\n")
        sys.exit(1)
    model = keras_load(path)
    print(f"  Mask model sẵn sàng: {Path(path).name}")
    return model


#  BƯỚC 3 – PHÂN LOẠI MASK
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


def predict_masks(
    frame: np.ndarray,
    faces: list,
    model,
    smoother: TemporalSmoother | None = None,
) -> list[Prediction]:
    """Phân loại mask cho từng khuôn mặt."""
    if not faces:
        if smoother is not None:
            smoother.update([], [])
        return []

    crops = []
    for face_box in faces:
        face = crop_face(frame, face_box, padding_ratio=CFG.FACE_PADDING_RATIO)
        if face is None or face.size == 0:
            face = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        crops.append(prepare_classifier_input(face))

    probs = [
        np.asarray(p, dtype=np.float32)
        for p in model.predict(np.array(crops, dtype=np.float32), verbose=0)
    ]
    if smoother is not None:
        probs = smoother.update(faces, probs)

    results = []
    for p in probs:
        label, confidence, is_mask = decode_prediction(p)
        results.append(Prediction(
            label=label,
            confidence=confidence,
            is_mask=is_mask,
        ))
    return results


def analyze_image_frame(frame: np.ndarray, face_net, mask_model) -> dict[str, Any]:
    """Chạy toàn bộ pipeline suy luận cho một frame ảnh tĩnh."""
    t0 = time.perf_counter()
    faces = detect_faces(frame, face_net)
    preds = predict_masks(frame, faces, mask_model)
    ms = (time.perf_counter() - t0) * 1000

    annotated = draw_results(frame.copy(), faces, preds)
    cv2.putText(
        annotated,
        f"{ms:.0f} ms",
        (annotated.shape[1] - 80, annotated.shape[0] - 10),
        FONT,
        0.45,
        YELLOW,
        1,
    )

    details = []
    for i, ((x1, y1, x2, y2, fc), pred) in enumerate(zip(faces, preds), 1):
        details.append(
            {
                "index": i,
                "label": pred.label,
                "confidence": pred.confidence,
                "is_mask": pred.is_mask,
                "face_confidence": float(fc),
                "box": (x1, y1, x2, y2),
            }
        )

    return {
        "annotated_frame": annotated,
        "faces": len(faces),
        "predictions": [(p.label, p.confidence) for p in preds],
        "elapsed_ms": ms,
        "details": details,
    }


#  BƯỚC 4 – VẼ KẾT QUẢ LÊN FRAME

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

#  CHẾ ĐỘ 1 – WEBCAM REALTIME
def run_webcam(face_net, mask_model) -> None:
    print("\n  WEBCAM MODE")
    print("  ─────────────────────────────────────")
    print("  [Q]  Thoát")
    print("  [S]  Chụp ảnh màn hình")
    print("  [P]  Tạm dừng / Tiếp tục")
    print("  ─────────────────────────────────────\n")

    cap = cv2.VideoCapture(CFG.CAMERA_ID)
    if not cap.isOpened():
        print(f" Không mở được camera {CFG.CAMERA_ID}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.CAMERA_HEIGHT)

    paused    = False
    prev_time = time.time()
    shot_idx  = 0
    smoother  = TemporalSmoother(
        alpha=CFG.SMOOTHING_ALPHA,
        max_distance=CFG.TRACK_MAX_DISTANCE,
        ttl=CFG.TRACK_TTL,
    )
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("  Không đọc được frame")
                break

            faces = detect_faces(frame, face_net)
            preds = predict_masks(frame, faces, mask_model, smoother=smoother)

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
    print("  Đã thoát webcam.")


# ══════════════════════════════════════════════════════════
#  CHẾ ĐỘ 2 – ẢNH ĐƠN LẺ
# ══════════════════════════════════════════════════════════
def run_image(path: str, face_net, mask_model,
              show: bool = True, save: bool = True) -> dict:
    """Nhận diện mask trong 1 ảnh tĩnh."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"  Không đọc được ảnh: {path}")
        return {}

    result = analyze_image_frame(frame, face_net, mask_model)
    annotated = result["annotated_frame"]

    # In kết quả
    print(f"\n  📷  {Path(path).name}")
    print(f"  Thời gian  : {result['elapsed_ms']:.0f} ms")
    print(f"  Khuôn mặt : {result['faces']}")
    for item in result["details"]:
        x1, y1, x2, y2 = item["box"]
        print(
            f"    [{item['index']}] {item['label']:8s} {item['confidence']:.0%}"
            f"  | face_conf={item['face_confidence']:.0%}"
            f"  box=({x1},{y1})→({x2},{y2})"
        )

    if save:
        out = Path("results") / f"pred_{Path(path).name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated)
        print(f" Lưu kết quả: {out}")

    if show:
        cv2.imshow(f"[Q] đóng – {Path(path).name}", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "file": path,
        "faces": result["faces"],
        "predictions": result["predictions"],
        "elapsed_ms": result["elapsed_ms"],
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
        print(f"  Không tìm thấy ảnh trong: {folder}")
        return

    print(f"\n  BATCH MODE – {len(images)} ảnh")
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
    print(f"  Hoàn tất batch processing")
    print(f"  Số ảnh đã xử lý   : {len(all_results)}")
    print(f"  Tổng khuôn mặt    : {total_mask + total_nomask}")
    print(f"  Đeo khẩu trang  : {total_mask}")
    print(f"  Không đeo        : {total_nomask}")
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
        "--mode", choices=["webcam", "image", "batch", "gui"], default=None,
        help="webcam  – realtime từ webcam\n"
             "image   – xử lý 1 ảnh, hoặc mở GUI nếu không truyền --input\n"
             "batch   – xử lý cả thư mục\n"
             "gui     – giao diện chọn ảnh tĩnh\n"
             "mac dinh – hien man hinh chon Webcam / Anh tinh",
    )
    parser.add_argument("--input",   default=None, help="Đường dẫn ảnh hoặc thư mục")
    parser.add_argument("--no-show", action="store_true",
                        help="Không hiển thị cửa sổ (headless / Colab)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║  FACE MASK – BƯỚC 3: DEMO APPLICATION   ║")
    print("╚══════════════════════════════════════════╝\n")

    mode = args.mode or choose_demo_mode()
    if mode is None:
        print("  Khong co che do nao duoc chon. Da thoat.")
        return

    # Load models
    face_net   = load_face_detector()
    print("  Face detector sẵn sàng (OpenCV DNN – Caffe SSD)")
    mask_model = load_mask_model()

    # Phân nhánh
    if mode == "webcam":
        run_webcam(face_net, mask_model)

    elif mode == "image":
        if args.input:
            run_image(args.input, face_net, mask_model,
                      show=not args.no_show, save=True)
        else:
            run_gui(face_net, mask_model)

    elif mode == "batch":
        if not args.input:
            parser.error("--mode batch cần --input <thư mục>")
        run_batch(args.input, face_net, mask_model)

    elif mode == "gui":
        run_gui(face_net, mask_model)


if __name__ == "__main__":
    main()
