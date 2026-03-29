import sys
from pathlib import Path

import numpy as np

# Add project root so tests can import config module
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import config


def test_processed_data_exists():
    """Check whether all 6 required .npy files exist."""
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
        assert file_path.exists(), f"Missing processed data file: {file}"


def test_data_shapes_and_values():
    """Check data integrity (shape and normalization)."""
    proc_dir = config.DATA_PROC

    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")

    # 1) Number of samples and labels should match.
    assert len(X_train) == len(y_train), "Image count and label count mismatch."

    # 2) Image shape should match config.
    expected_shape = config.CFG.IMAGE_SIZE + (3,)
    assert (
        X_train.shape[1:] == expected_shape
    ), f"Invalid image shape. Got {X_train.shape[1:]}, expected {expected_shape}"

    # 3) Pixel values should be normalized in [0, 1].
    assert (
        np.max(X_train) <= 1.0 and np.min(X_train) >= 0.0
    ), "Data is not normalized to range [0, 1]."
