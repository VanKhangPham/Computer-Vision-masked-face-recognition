"""
Người phụ trách: TV 1 (Data)
Chức năng: Đọc ảnh từ data/raw, tiền xử lý, chia tập dữ liệu và lưu vào data/processed
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Thêm thư mục gốc vào hệ thống để import được config.py
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import config


def load_and_preprocess_data():
    raw_dir = config.DATA_RAW
    classes = config.CFG.CLASSES
    img_width, img_height = config.CFG.IMAGE_SIZE

    data = []
    labels = []

    print("=" * 50)
    print(" 🛠 BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU (DATA PREPARATION)")
    print("=" * 50)

    for cls_name in classes:
        folder_path = raw_dir / cls_name

        if not folder_path.exists():
            print(f"❌ Lỗi: Không tìm thấy thư mục {folder_path}!")
            print("Vui lòng tải dataset và đặt vào đúng thư mục data/raw/")
            sys.exit(1)

        label = classes.index(cls_name)
        img_list = os.listdir(folder_path)

        print(f" -> Đang đọc thư mục '{cls_name}' ({len(img_list)} ảnh)...")

        for img_name in img_list:
            img_path = folder_path / img_name
            try:
                # Đọc bằng OpenCV
                img_array = cv2.imread(str(img_path))
                if img_array is not None:
                    # Chuyển BGR sang RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    # Resize theo cấu hình
                    resized_array = cv2.resize(img_array, (img_width, img_height))

                    data.append(resized_array)
                    labels.append(label)
            except Exception as e:
                pass  # Bỏ qua các file không hợp lệ

    # Chuyển thành Numpy Array và Chuẩn hóa (0-255 -> 0-1)
    print("\n⏳ Đang chuẩn hóa ma trận pixel...")
    data = np.array(data, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)

    return data, labels


def split_and_save_data(data, labels):
    proc_dir = config.DATA_PROC
    cfg = config.CFG

    print("\n🔀 Đang phân chia tập dữ liệu theo Config...")

    # Tính toán tỷ lệ tập Test & Val so với tổng thể (Mặc định 0.1 + 0.1 = 0.2)
    val_test_ratio = cfg.VAL_RATIO + cfg.TEST_RATIO

    # Lần cắt 1: Tách tập Train (80%) và tập Tạm (20% gồm Val + Test)
    # Stratify=labels giúp cân bằng số lượng mask/no_mask trong mỗi tập
    X_train, X_temp, y_train, y_temp = train_test_split(
        data,
        labels,
        test_size=val_test_ratio,
        random_state=cfg.RANDOM_SEED,
        stratify=labels,
    )

    # Lần cắt 2: Chia tập Tạm thành Val (50% của 20% = 10%) và Test (10%)
    test_ratio_relative = cfg.TEST_RATIO / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio_relative,
        random_state=cfg.RANDOM_SEED,
        stratify=y_temp,
    )

    print(f" ✔ Tập Train (Huấn luyện): {len(X_train)} ảnh")
    print(f" ✔ Tập Val   (Đánh giá)  : {len(X_val)} ảnh")
    print(f" ✔ Tập Test  (Kiểm thử)  : {len(X_test)} ảnh")

    # Lưu dữ liệu vào data/processed
    print("\n💾 Đang lưu ma trận dữ liệu (.npy) cho Người 2...")
    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "y_train.npy", y_train)
    np.save(proc_dir / "X_val.npy", X_val)
    np.save(proc_dir / "y_val.npy", y_val)
    np.save(proc_dir / "X_test.npy", X_test)
    np.save(proc_dir / "y_test.npy", y_test)

    print(f"\n✅ HOÀN TẤT NHIỆM VỤ NGƯỜI 1! Dữ liệu đã sẵn sàng tại: {proc_dir}")


if __name__ == "__main__":
    # Luồng chạy chính
    X, y = load_and_preprocess_data()
    split_and_save_data(X, y)
