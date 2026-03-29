"""
Member: TV 2
Task: build, train, evaluate, and save face-mask classification model.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Avoid UnicodeEncodeError on some Windows terminals.
for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CFG, LOGS_DIR, MODELS_DIR, PLOTS_DIR
from src.data_preparation import create_generators


def _num_samples(gen) -> int:
    return int(getattr(gen, "samples", getattr(gen, "n", 0)))


def build_model() -> tuple[Model, Model]:
    """
    Backbone: MobileNetV2 pretrained on ImageNet (initially frozen).
    Head: AvgPool -> Flatten -> Dense -> BN -> Dropout -> Dense -> Dropout -> Softmax.
    """
    print("\nBUILD MODEL")
    print("-" * 50)
    print(f"Backbone : {CFG.BACKBONE}")
    print(f"Input    : {CFG.IMAGE_SIZE[0]}x{CFG.IMAGE_SIZE[1]}x3")
    print(f"Classes  : {CFG.CLASSES}")

    base = MobileNetV2(
        weights=CFG.PRETRAINED_WEIGHTS,
        include_top=False,
        input_tensor=Input(shape=(*CFG.IMAGE_SIZE, 3)),
    )
    base.trainable = False

    x = base.output
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(CFG.DENSE_UNITS, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(CFG.DROPOUT_1)(x)
    x = Dense(CFG.DENSE_UNITS // 2, activation="relu")(x)
    x = Dropout(CFG.DROPOUT_2)(x)
    out = Dense(len(CFG.CLASSES), activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)

    head_params = model.count_params() - base.count_params()
    print(f"Backbone params: {base.count_params():,} (frozen)")
    print(f"Head params    : {head_params:,} (trainable)")
    print(f"Total params   : {model.count_params():,}")

    return model, base


def _compile(model: Model, lr: float) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _get_callbacks(phase: int) -> list:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    return [
        ModelCheckpoint(
            filepath=str(MODELS_DIR / f"best_phase{phase}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=CFG.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=str(LOGS_DIR / f"phase{phase}"), histogram_freq=0),
    ]


def train_phase1(model, base, train_gen, val_gen, epochs: int):
    """
    Phase 1: train only custom head.
    """
    print("\n" + "=" * 50)
    print("PHASE 1: TRAIN CUSTOM HEAD")
    print(f"LR = {CFG.INIT_LR} | Epochs = {epochs}")
    print("=" * 50)

    base.trainable = False
    model = _compile(model, CFG.INIT_LR)

    return model.fit(
        train_gen,
        steps_per_epoch=max(1, _num_samples(train_gen) // CFG.BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, _num_samples(val_gen) // CFG.BATCH_SIZE),
        epochs=epochs,
        callbacks=_get_callbacks(1),
    )


def train_phase2(model, base, train_gen, val_gen, epochs: int):
    """
    Phase 2: unfreeze tail of backbone and fine-tune.
    """
    print("\n" + "=" * 50)
    print("PHASE 2: FINE-TUNE BACKBONE")
    print(f"Unfreeze from layer {CFG.FINE_TUNE_AT} | LR = {CFG.FINE_TUNE_LR}")
    print("=" * 50)

    base.trainable = True
    for layer in base.layers[: CFG.FINE_TUNE_AT]:
        layer.trainable = False

    n_trainable = sum(1 for layer in base.layers if layer.trainable)
    print(f"Backbone trainable layers: {n_trainable}/{len(base.layers)}")

    model = _compile(model, CFG.FINE_TUNE_LR)

    return model.fit(
        train_gen,
        steps_per_epoch=max(1, _num_samples(train_gen) // CFG.BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, _num_samples(val_gen) // CFG.BATCH_SIZE),
        epochs=epochs,
        callbacks=_get_callbacks(2),
    )


def evaluate_model(model: Model, test_gen) -> dict:
    print("\n" + "=" * 50)
    print("EVALUATE ON TEST SET")
    print("=" * 50)

    test_samples = _num_samples(test_gen)
    if test_samples <= 0:
        raise ValueError("Test split is empty.")

    test_gen.reset()
    steps = int(np.ceil(test_samples / CFG.BATCH_SIZE))
    probs = model.predict(test_gen, steps=steps, verbose=1)
    y_pred = np.argmax(probs, axis=1)[:test_samples]
    y_true = np.asarray(test_gen.classes)[:test_samples]

    class_indices = getattr(
        test_gen, "class_indices", {name: i for i, name in enumerate(CFG.CLASSES)}
    )
    names = [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]

    test_gen.reset()
    loss, acc, prec, rec = model.evaluate(test_gen, verbose=0)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print(f"Test Loss     : {loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=names))

    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": probs,
        "class_names": names,
    }


def _merge_history(h1, h2=None) -> tuple[dict, int]:
    keys = ["loss", "val_loss", "accuracy", "val_accuracy"]
    merged = {k: h1.history[k] + (h2.history[k] if h2 else []) for k in keys}
    split = len(h1.history["loss"])
    return merged, split


def plot_training_history(h1, h2=None) -> None:
    merged, split = _merge_history(h1, h2)
    epochs = range(1, len(merged["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    ax1.plot(epochs, merged["loss"], "b-o", ms=3, label="Train")
    ax1.plot(epochs, merged["val_loss"], "r-o", ms=3, label="Val")
    if h2:
        ax1.axvline(split, color="#aaa", ls="--", lw=1, label=f"Fine-tune at {split + 1}")
    ax1.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in merged["accuracy"]], "b-o", ms=3, label="Train")
    ax2.plot(
        epochs,
        [a * 100 for a in merged["val_accuracy"]],
        "r-o",
        ms=3,
        label="Val",
    )
    if h2:
        ax2.axvline(split, color="#aaa", ls="--", lw=1)
    ax2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=[0, 105])
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "training_history.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_confusion_matrix(results: dict) -> None:
    cm = confusion_matrix(results["y_true"], results["y_pred"])
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    kw = dict(
        annot=True,
        cmap="Blues",
        xticklabels=results["class_names"],
        yticklabels=results["class_names"],
    )
    sns.heatmap(cm, fmt="d", ax=ax1, **kw)
    sns.heatmap(cm_norm, fmt=".2%", ax=ax2, **kw)

    for ax, title in zip([ax1, ax2], ["Count", "Ratio"]):
        ax.set(title=title, ylabel="Actual", xlabel="Predicted")

    plt.tight_layout()
    path = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_roc_curve(results: dict) -> None:
    n = len(results["class_names"])
    y_bin = np.eye(n)[results["y_true"]]

    plt.figure(figsize=(7, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, (name, color) in enumerate(zip(results["class_names"], colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], results["y_prob"][:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = PLOTS_DIR / "roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_sample_predictions(model: Model, test_gen, n: int = 8) -> None:
    test_gen.reset()
    batch_x, batch_y = next(test_gen)
    batch_x = batch_x[:n]
    batch_y = batch_y[:n]

    preds = model.predict(batch_x, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(batch_y, axis=1)
    names = list(test_gen.class_indices.keys())

    fig, axes = plt.subplots(2, n // 2, figsize=(n * 2, 6))
    fig.suptitle("Sample Predictions", fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        img = batch_x[i]
        ax.imshow(img)
        ax.axis("off")

        pred = names[pred_labels[i]]
        true = names[true_labels[i]]
        conf = preds[i][pred_labels[i]]
        ok = pred_labels[i] == true_labels[i]
        tag = "[OK]" if ok else "[MISS]"
        color = "green" if ok else "red"

        ax.set_title(
            f"{tag} Pred: {pred.split('_')[-1]}\n"
            f"True: {true.split('_')[-1]} ({conf:.0%})",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    plt.tight_layout()
    path = PLOTS_DIR / "sample_predictions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Face Mask - Step 2: Train model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=str(__import__("config").DATA_PROC),
        help="Processed data folder (supports train/val/test or X_*.npy + y_*.npy)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CFG.EPOCHS_PHASE1,
        help="Number of epochs for phase 1",
    )
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help="Skip phase 2 fine-tuning",
    )
    args = parser.parse_args()

    CFG.EPOCHS_PHASE1 = args.epochs

    print("=" * 42)
    print(" FACE MASK - STEP 2: MODEL TRAINING")
    print("=" * 42)

    gpus = tf.config.list_physical_devices("GPU")
    device = f"GPU ({len(gpus)} device[s])" if gpus else "CPU (Colab recommended)"
    print(f"\nDevice: {device}")

    train_gen, val_gen, test_gen = create_generators(Path(args.data))

    model, base = build_model()

    h1 = train_phase1(model, base, train_gen, val_gen, CFG.EPOCHS_PHASE1)
    h2 = None
    if not args.no_finetune:
        h2 = train_phase2(model, base, train_gen, val_gen, CFG.EPOCHS_PHASE2)

    print("\nVISUALIZE TRAINING")
    print("-" * 50)
    plot_training_history(h1, h2)

    results = evaluate_model(model, test_gen)
    plot_confusion_matrix(results)
    plot_roc_curve(results)
    plot_sample_predictions(model, test_gen)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_path = MODELS_DIR / "mask_detector_final.keras"
    model.save(str(final_path))
    print(f"\nSaved final model: {final_path}")
    print("\nTraining completed. You can run app.py now.\n")


if __name__ == "__main__":
    main()
