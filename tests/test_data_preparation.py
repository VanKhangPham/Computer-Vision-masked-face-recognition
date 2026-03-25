import os
import numpy as np
from pathlib import Path
import sys

# Thêm thư mục gốc vào path để nhận diện module
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import config


def test_processed_data_exists():
    """Kiểm tra xem cả 6 file .npy đã được tạo ra đầy đủ chưa"""
    proc_dir = config.DATA_PROC
    expected_files = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
    ]

    for file in expected_files:
        file_path = proc_dir / file
        assert file_path.exists(), f"❌ Thiếu file dữ liệu: {file}"


def test_data_shapes_and_values():
    """Kiểm tra tính toàn vẹn của dữ liệu (kích thước, chuẩn hóa)"""
    proc_dir = config.DATA_PROC

    # Load thử tập Train
    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")

    # 1. Số lượng ảnh và nhãn phải khớp nhau
    assert len(X_train) == len(y_train), "❌ Số lượng ảnh và nhãn không khớp!"

    # 2. Kích thước ảnh phải đúng chuẩn (224, 224, 3) theo config
    expected_shape = config.CFG.IMAGE_SIZE + (3,)
    assert (
        X_train.shape[1:] == expected_shape
    ), f"❌ Kích thước ảnh sai. Đang là {X_train.shape[1:]}, mong đợi {expected_shape}"

    # 3. Dữ liệu phải được chuẩn hóa trong khoảng [0, 1]
    assert (
        np.max(X_train) <= 1.0 and np.min(X_train) >= 0.0
    ), "❌ Dữ liệu chưa được chuẩn hóa (Normalization) đúng cách!"
