# Hướng dẫn bắt đầu nhanh cho thành viên mới

> Đọc file này **trước tiên** khi bắt đầu tham gia dự án.

---





## Bước 1 – Cài Extensions được đề xuất

Khi mở dự án, VS Code sẽ hiện thông báo:

> *"This workspace has extension recommendations. Would you like to install them?"*

Nhấn **Install All** để cài toàn bộ extensions cần thiết.

Hoặc mở thủ công: `Ctrl+Shift+P` → gõ `Show Recommended Extensions`.

---

## Bước 2 – Tạo môi trường ảo & cài thư viện

Dùng **Terminal tích hợp** trong VS Code (`Ctrl + \``):

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Hoặc dùng Task có sẵn: `Ctrl+Shift+P` → `Run Task` → ** Setup │ Cài đặt toàn bộ**.

---

## Bước 3 – Chọn Python Interpreter

Nhấn `Ctrl+Shift+P` → gõ `Python: Select Interpreter` → chọn `.venv`.

>  Góc dưới bên trái VS Code sẽ hiển thị phiên bản Python đang dùng.

---

## Bước 4 – Kiểm tra cài đặt

```bash
# Chạy config để xác nhận mọi thứ OK
python config.py
```

Kết quả mong đợi:
```
==================================================
  CẤU HÌNH DỰ ÁN FACE MASK DETECTION
==================================================
  ROOT_DIR   : /path/to/face_mask_detection
  ...
 Config OK!
```

---

## Bước 5 – Chạy dự án theo nhiệm vụ

### Cách 1: Dùng Run & Debug (F5)
Nhấn `F5` hoặc vào tab **Run and Debug** (Ctrl+Shift+D), chọn cấu hình phù hợp:

```
 Người 1 │ Chuẩn bị dữ liệu (đầy đủ)
 Người 2 │ Train model (20 epochs)
 Người 3 │ Chạy Webcam Demo
 Debug file đang mở
 Chạy toàn bộ Tests
```

### Cách 2: Dùng Tasks (Ctrl+Shift+B)
`Ctrl+Shift+P` → `Run Task` → chọn tác vụ muốn chạy.

---

## Cấu trúc thư mục

```
face_mask_detection/
│
├── .vscode/                    ← Cấu hình VS Code (KHÔNG sửa)
│   ├── settings.json           │  - Format, linting, Python path
│   ├── launch.json             │  - Debug configurations
│   ├── tasks.json              │  - One-click tasks
│   └── extensions.json         │  - Extensions đề xuất
│
├── src/                        ← MÃ NGUỒN CHÍNH
│   ├── data_preparation.py     │  👤 Người 1 phụ trách
│   ├── train_model.py          │  👤 Người 2 phụ trách
│   └── app.py                  │  👤 Người 3 phụ trách
│
├── tests/                      ← Unit tests (chạy trước khi commit)
│   ├── test_data_preparation.py
│   └── test_model.py
│
├── data/
│   ├── raw/                    ← Đặt dataset Kaggle vào đây
│   │   ├── with_mask/
│   │   └── without_mask/
│   └── processed/              ← Tự tạo sau khi chạy TV 1
│
├── models/                     ← File .keras sau training
├── notebooks/
│   └── train_colab.ipynb       ← Notebook Google Colab
├── results/
│   ├── plots/                  ← Biểu đồ kết quả
│   └── screenshots/            ← Ảnh chụp từ webcam
│
├── config.py                   ←  Cấu hình trung tâm (import từ đây)
├── requirements.txt
├── pytest.ini
├── .gitignore
└── SETUP.md                    ← File này
```

---

## Quy tắc làm việc nhóm

### Git workflow
```bash
# Trước khi bắt đầu làm việc mỗi ngày()
git pull origin main

# Tạo branch riêng và tự động chuyển tới nhánh đó(làm trên github cho chăc nếu chưa quen)
git checkout -b nguoi1/data-augmentation

# Commit thường xuyên, message rõ ràng
git add src/data_preparation.py
git commit -m "(tính năng hoặc phần nào đó mà vừa làm, làm sao hiểu là đc)"

# Push và tạo Pull Request
git push origin nguoi1/data-augmentation
```

### Quy tắc code
- **Không hardcode** đường dẫn – dùng `config.py`
- **Chạy tests** trước khi commit: `pytest tests/ -v`
- **Docstring** cho mỗi hàm mới (dùng extension AutoDocstring)
- **Comment bằng tiếng Việt** cho phần logic phức tạp(dễ hiểu là đc)

---

## Gỡ lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `ModuleNotFoundError: tensorflow` | Chưa activate venv | `.venv\Scripts\activate` |
| `No module named 'src'` | PYTHONPATH chưa đặt | Dùng launch.json thay vì chạy thủ công |
| Camera không mở | camera_id sai | Thử `--camera 1` hoặc `2` |
| `FileNotFoundError: mask_detector_final.keras` | Chưa train | Chạy Người 2 trước |
| GPU không được nhận | CUDA/cuDNN chưa cài | Chạy trên Google Colab |

---

## Liên hệ nhóm

> Điền thông tin nhóm vào đây:

| Thành viên | Nhiệm vụ | GitHub |
|------------|----------|--------|
| TV 1    | Data     | @...   |
| TV 2    | Model    | @...   |
| TV 3    | App      | @...   |
