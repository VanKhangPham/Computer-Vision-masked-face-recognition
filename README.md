# Computer Vision - Masked Face Recognition

Dự án nhận diện đeo khẩu trang theo thời gian thực bằng webcam, ảnh đơn hoặc batch ảnh.
Pipeline gồm 3 phần chính:

1. Chuẩn bị dữ liệu (`src/data_preparation.py`)
2. Huấn luyện mô hình (`src/train_model.py`)
3. Chạy ứng dụng demo (`src/app.py`)

## Tính năng chính

- Phân loại `with_mask` / `without_mask` bằng mô hình CNN (MobileNetV2 transfer learning).
- Hỗ trợ huấn luyện 2 giai đoạn: train head + fine-tune backbone.
- Demo realtime bằng webcam với HUD, FPS, chụp ảnh nhanh.
- Chạy suy luận cho 1 ảnh hoặc cả thư mục ảnh.
- Sinh biểu đồ đánh giá: history, confusion matrix, ROC, sample predictions.
- Unit test cơ bản cho dữ liệu đã xử lý.

## Công nghệ sử dụng

- Python
- TensorFlow / Keras
- OpenCV
- scikit-learn
- NumPy, Matplotlib, Seaborn

## Cấu trúc thư mục

```text
Computer-Vision-masked-face-recognition/
├─ src/
│  ├─ data_preparation.py
│  ├─ train_model.py
│  └─ app.py
├─ tests/
│  └─ test_data_preparation.py
├─ data/
│  ├─ raw/
│  │  ├─ with_mask/
│  │  └─ without_mask/
│  └─ processed/
├─ models/
├─ results/
│  ├─ logs/
│  ├─ plots/
│  └─ screenshots/
├─ config.py
├─ requirements.txt
├─ pytest.ini
└─ SETUP.md
```

## Cài đặt nhanh

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kiểm tra cấu hình:

```bash
python config.py
```

## Chuẩn bị dữ liệu

Đặt dữ liệu gốc theo cấu trúc:

```text
data/raw/
├─ with_mask/
└─ without_mask/
```

Sau đó chạy:

```bash
python src/data_preparation.py
```

Script sẽ tạo các file:

- `data/processed/X_train.npy`
- `data/processed/y_train.npy`
- `data/processed/X_val.npy`
- `data/processed/y_val.npy`
- `data/processed/X_test.npy`
- `data/processed/y_test.npy`

## Huấn luyện mô hình

Chạy training mặc định:

```bash
python src/train_model.py
```

Train nhanh để kiểm thử:

```bash
python src/train_model.py --epochs 5 --no-finetune
```

Output chính:

- `models/mask_detector_final.keras`
- `models/best_phase1.keras`
- `models/best_phase2.keras`
- `results/plots/*.png`
- `results/logs/phase*/`

### Lưu ý quan trọng về format dữ liệu train

`src/train_model.py` đang dùng `flow_from_directory`, tức kỳ vọng dữ liệu theo cấu trúc:

```text
data/processed/
├─ train/
│  ├─ with_mask/
│  └─ without_mask/
├─ val/
│  ├─ with_mask/
│  └─ without_mask/
└─ test/
   ├─ with_mask/
   └─ without_mask/
```

Trong khi `src/data_preparation.py` hiện tại sinh file `.npy`.  
Nếu bạn train bằng `train_model.py`, cần chuẩn bị thêm cấu trúc thư mục `train/val/test` như trên hoặc cập nhật lại script train cho phù hợp với `.npy`.

## Chạy ứng dụng demo

### Webcam realtime

```bash
python src/app.py --mode webcam
```

Phím tắt:

- `Q`: thoát
- `S`: chụp ảnh màn hình (lưu vào `results/screenshots/`)
- `P`: tạm dừng/tiếp tục

### Ảnh đơn

```bash
python src/app.py --mode image --input path/to/image.jpg
```

### Batch thư mục ảnh

```bash
python src/app.py --mode batch --input path/to/folder
```

Lưu ý:

- `app.py` tự tải file face detector (`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`) vào thư mục `models/` nếu chưa có.
- Cần có `models/mask_detector_final.keras` trước khi chạy app.

## Chạy test

```bash
pytest tests/ -v
```

## Cấu hình trung tâm

Toàn bộ tham số được quản lý ở `config.py`, ví dụ:

- Kích thước ảnh: `CFG.IMAGE_SIZE`
- Batch size: `CFG.BATCH_SIZE`
- Số epoch: `CFG.EPOCHS_PHASE1`, `CFG.EPOCHS_PHASE2`
- Ngưỡng nhận diện: `CFG.FACE_CONFIDENCE`, `CFG.MASK_THRESHOLD`
- Camera ID: `CFG.CAMERA_ID`

## Xử lý lỗi thường gặp

- `ModuleNotFoundError: tensorflow`: chưa kích hoạt môi trường ảo hoặc chưa cài dependencies.
- `FileNotFoundError: mask_detector_final.keras`: chưa train model.
- Không mở được webcam: thử đổi `CFG.CAMERA_ID` trong `config.py` (0, 1, 2...).
- Lỗi thiếu dữ liệu train khi chạy `train_model.py`: kiểm tra lại format thư mục `data/processed/train|val|test`.

## Gợi ý cải tiến tiếp theo

- Đồng bộ `data_preparation.py` và `train_model.py` về cùng một format dữ liệu.
- Thêm test cho `train_model.py` và `app.py`.
- Thêm script đánh giá/benchmark tự động và lưu metrics vào file JSON/CSV.
