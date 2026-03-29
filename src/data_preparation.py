"""
Member: TV 1 (Data)
Task: read images from data/raw, preprocess, split datasets, and save to data/processed.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Add project root for importing config.py
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import config
from config import CFG, DATA_PROC


def _annotate_numpy_generator(gen, labels: np.ndarray):
    """
    Add metadata so numpy generators behave similarly to directory generators.
    """
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    gen.samples = int(labels.shape[0])
    gen.classes = labels
    gen.class_indices = {name: idx for idx, name in enumerate(CFG.CLASSES)}
    return gen


def _load_npy_split(processed_dir: Path, split: str):
    x_path = processed_dir / f"X_{split}.npy"
    y_path = processed_dir / f"y_{split}.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing .npy files for split '{split}' in {processed_dir}"
        )

    # Use mmap_mode to reduce peak memory pressure.
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)

    if len(X) == 0 or len(y) == 0:
        raise ValueError(f"Split '{split}' is empty.")
    if len(X) != len(y):
        raise ValueError(
            f"Split '{split}' mismatch: images={len(X)} labels={len(y)}"
        )

    y = y.astype(np.int32, copy=False).reshape(-1)
    if np.min(y) < 0 or np.max(y) >= len(CFG.CLASSES):
        raise ValueError(
            f"Split '{split}' labels out of range [0, {len(CFG.CLASSES) - 1}]"
        )

    return X, y


def _flow_from_numpy(aug, X, y, shuffle: bool):
    y_categorical = to_categorical(y, num_classes=len(CFG.CLASSES))
    gen = aug.flow(
        X,
        y_categorical,
        batch_size=CFG.BATCH_SIZE,
        shuffle=shuffle,
        seed=CFG.RANDOM_SEED,
    )
    return _annotate_numpy_generator(gen, y)


def create_generators(processed_dir: Path = DATA_PROC):
    """
    Create train/val/test generators.

    Supported formats:
    1) Directory format: processed/train|val|test/<class>/*
    2) Numpy format: X_train.npy, y_train.npy, ...
    """
    processed_dir = Path(processed_dir)

    aug_kwargs = dict(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    # Directory image data requires rescale; npy data is already normalized.
    train_aug_dir = ImageDataGenerator(rescale=1.0 / 255.0, **aug_kwargs)
    eval_aug_dir = ImageDataGenerator(rescale=1.0 / 255.0)
    train_aug_npy = ImageDataGenerator(**aug_kwargs)
    eval_aug_npy = ImageDataGenerator()

    dir_splits = [processed_dir / s for s in ("train", "val", "test")]
    npy_files = [
        processed_dir / f"{prefix}_{split}.npy"
        for split in ("train", "val", "test")
        for prefix in ("X", "y")
    ]

    if all(p.exists() for p in dir_splits):
        train_gen = train_aug_dir.flow_from_directory(
            str(processed_dir / "train"),
            target_size=CFG.IMAGE_SIZE,
            batch_size=CFG.BATCH_SIZE,
            class_mode="categorical",
            classes=CFG.CLASSES,
            shuffle=True,
            seed=CFG.RANDOM_SEED,
        )
        val_gen = eval_aug_dir.flow_from_directory(
            str(processed_dir / "val"),
            target_size=CFG.IMAGE_SIZE,
            batch_size=CFG.BATCH_SIZE,
            class_mode="categorical",
            classes=CFG.CLASSES,
            shuffle=False,
            seed=CFG.RANDOM_SEED,
        )
        test_gen = eval_aug_dir.flow_from_directory(
            str(processed_dir / "test"),
            target_size=CFG.IMAGE_SIZE,
            batch_size=CFG.BATCH_SIZE,
            class_mode="categorical",
            classes=CFG.CLASSES,
            shuffle=False,
            seed=CFG.RANDOM_SEED,
        )
    elif all(p.exists() for p in npy_files):
        X_train, y_train = _load_npy_split(processed_dir, "train")
        X_val, y_val = _load_npy_split(processed_dir, "val")
        X_test, y_test = _load_npy_split(processed_dir, "test")

        train_gen = _flow_from_numpy(train_aug_npy, X_train, y_train, True)
        val_gen = _flow_from_numpy(eval_aug_npy, X_val, y_val, False)
        test_gen = _flow_from_numpy(eval_aug_npy, X_test, y_test, False)
    else:
        raise FileNotFoundError(
            "No valid processed data found. Expected either:\n"
            "- directory format train/val/test/<class>\n"
            "- numpy format X_*.npy and y_*.npy"
        )

    for split_name, gen in (("train", train_gen), ("val", val_gen), ("test", test_gen)):
        samples = int(getattr(gen, "samples", getattr(gen, "n", 0)))
        if samples <= 0:
            raise ValueError(
                f"Split '{split_name}' has 0 samples. Check data in {processed_dir}"
            )

    return train_gen, val_gen, test_gen


def load_and_preprocess_data():
    raw_dir = config.DATA_RAW
    classes = config.CFG.CLASSES
    img_width, img_height = config.CFG.IMAGE_SIZE

    data = []
    labels = []

    print("=" * 50)
    print(" START DATA PREPARATION")
    print("=" * 50)

    for cls_name in classes:
        folder_path = raw_dir / cls_name

        if not folder_path.exists():
            print(f"ERROR: Missing folder {folder_path}")
            print("Please put dataset into data/raw/<class_name>/")
            sys.exit(1)

        label = classes.index(cls_name)
        img_list = os.listdir(folder_path)

        print(f"Reading folder '{cls_name}' ({len(img_list)} images)...")

        for img_name in img_list:
            img_path = folder_path / img_name
            try:
                img_array = cv2.imread(str(img_path))
                if img_array is None:
                    continue

                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                resized_array = cv2.resize(img_array, (img_width, img_height))

                data.append(resized_array)
                labels.append(label)
            except Exception:
                # Skip invalid files.
                pass

    print("\nNormalizing pixel values...")
    data = np.array(data, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)

    return data, labels


def split_and_save_data(data, labels):
    proc_dir = config.DATA_PROC
    cfg = config.CFG

    print("\nSplitting dataset by config...")
    val_test_ratio = cfg.VAL_RATIO + cfg.TEST_RATIO

    X_train, X_temp, y_train, y_temp = train_test_split(
        data,
        labels,
        test_size=val_test_ratio,
        random_state=cfg.RANDOM_SEED,
        stratify=labels,
    )

    test_ratio_relative = cfg.TEST_RATIO / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio_relative,
        random_state=cfg.RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples  : {len(X_val)}")
    print(f"Test samples : {len(X_test)}")

    print("\nSaving .npy files...")
    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "y_train.npy", y_train)
    np.save(proc_dir / "X_val.npy", X_val)
    np.save(proc_dir / "y_val.npy", y_val)
    np.save(proc_dir / "X_test.npy", X_test)
    np.save(proc_dir / "y_test.npy", y_test)

    print(f"\nDone. Processed data saved in: {proc_dir}")


if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    split_and_save_data(X, y)
