import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root so tests can import config module
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import config


def _has_npy_format(proc_dir: Path) -> bool:
    expected_files = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
    ]
    return all((proc_dir / file).exists() for file in expected_files)


def _has_directory_format(proc_dir: Path) -> bool:
    return all((proc_dir / split).exists() for split in ("train", "val", "test"))


def test_processed_data_exists():
    """Check whether processed data exists in npy or directory format."""
    proc_dir = config.DATA_PROC
    assert _has_npy_format(proc_dir) or _has_directory_format(
        proc_dir
    ), "No processed dataset found."

    if _has_directory_format(proc_dir):
        for split in ("train", "val", "test"):
            for class_name in config.CFG.CLASSES:
                class_dir = proc_dir / split / class_name
                assert class_dir.exists(), f"Missing processed class folder: {class_dir}"
                assert any(class_dir.iterdir()), f"Processed class folder is empty: {class_dir}"


def test_data_shapes_and_values():
    """Check data integrity for the processed dataset."""
    proc_dir = config.DATA_PROC
    expected_shape = config.CFG.IMAGE_SIZE + (3,)

    if _has_npy_format(proc_dir):
        X_train = np.load(proc_dir / "X_train.npy")
        y_train = np.load(proc_dir / "y_train.npy")

        assert len(X_train) == len(y_train), "Image count and label count mismatch."
        assert (
            X_train.shape[1:] == expected_shape
        ), f"Invalid image shape. Got {X_train.shape[1:]}, expected {expected_shape}"
        assert (
            np.max(X_train) <= 1.0 and np.min(X_train) >= 0.0
        ), "Data is not normalized to range [0, 1]."
        return

    sample_file = next((proc_dir / "train" / config.CFG.CLASSES[0]).iterdir())
    image = cv2.imread(str(sample_file))

    assert image is not None, f"Unable to read processed image: {sample_file}"
    assert image.shape == expected_shape, f"Invalid image shape. Got {image.shape}, expected {expected_shape}"
    assert image.dtype == np.uint8, f"Expected uint8 processed image, got {image.dtype}"
    assert np.max(image) <= 255 and np.min(image) >= 0, "Processed image pixel range is invalid."
