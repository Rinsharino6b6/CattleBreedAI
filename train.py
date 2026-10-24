import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent))

from src.config import (
    DATASET_DIR, MODEL_PATH, CLASS_NAMES_PATH, TRAINING_HISTORY_PATH,
    OUTPUTS_DIR, IMAGE_SIZE, BATCH_SIZE, INITIAL_EPOCHS, FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE, FINE_TUNE_LEARNING_RATE
)
from src.data_utils import create_data_generators, scan_dataset_structure
from src.model import build_transfer_model, unfreeze_model_for_finetuning, build_data_augmentation_layers

def check_hardware_acceleration():
    """Detect and display GPU/CPU availability at startup."""
    gpus = tf.config.list_physical_devices('GPU')
    print("=" * 60)
    print("           HARDWARE & COMPUTATIONAL SETUP")
    print("=" * 60)
    if gpus:
        print(f"GPU Available : YES ({len(gpus)} GPU(s) detected)")
        for gpu in gpus:
            print(f"  - Device    : {gpu.name}")
    else:
        print("GPU Available : NO (Running training pipeline on CPU)")
    print(f"TensorFlow Ver: {tf.__version__}\n")

def run_training_pipeline():
    """Execute complete training and fine-tuning pipeline."""
    check_hardware_acceleration()

    print("=" * 60)
    print("         STEP 1: LOADING & VALIDATING DATASET")
    print("=" * 60)
    summary = scan_dataset_structure(DATASET_DIR)
    if not summary["exists"] or summary["total_valid_images"] == 0:
        print("[ERROR] No dataset found in dataset/ directory!")
        print("Please add breed class directories inside dataset/train/ or dataset/")
        print("Refer to README.md for instructions.")
        sys.exit(1)

    train_ds, val_ds, test_ds, class_names = create_data_generators(
        DATASET_DIR, IMAGE_SIZE, BATCH_SIZE
    )

    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[SAVED] Class mapping saved to: {CLASS_NAMES_PATH}\n")

    if train_ds is None:
        print("[ERROR] Failed to load training dataset generator.")
        sys.exit(1)

    augmentation_layers = build_data_augmentation_layers()
    train_ds_aug = train_ds.map(
        lambda x, y: (augmentation_layers(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    print("=" * 60)
    print("         STEP 2: BUILDING EFFICIENTNETB0 MODEL")
    print("=" * 60)
    model = build_transfer_model(
        num_classes=num_classes,
        base_architecture="EfficientNetB0",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy" if val_ds else "accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy" if val_ds else "accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds else "loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\n" + "=" * 60)
    print("    STEP 3: STAGE 1 - TRAINING CLASSIFICATION HEAD")
    print("=" * 60)
    history_stage1 = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks
    )

    print("\n" + "=" * 60)
    print("    STEP 4: STAGE 2 - FINE-TUNING CONVOLUTIONAL LAYERS")
    print("=" * 60)
    model = unfreeze_model_for_finetuning(
        model,
        unfreeze_from_layer=-30,
        learning_rate=FINE_TUNE_LEARNING_RATE
    )

    history_stage2 = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(history_stage1.history["accuracy"]),
        callbacks=callbacks
    )

    combined_history = {}
    for key in history_stage1.history:
        combined_history[key] = history_stage1.history[key] + history_stage2.history.get(key, [])

    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"[SAVED] Training history saved to: {TRAINING_HISTORY_PATH}")

    plot_and_save_curves(combined_history)

    print("\n" + "=" * 60)
    print(" [SUCCESS] MODEL TRAINING & FINE-TUNING COMPLETED!")
    print(f" Best model checkpoint: {MODEL_PATH}")
    print("=" * 60)

def plot_and_save_curves(history: dict):
    """Plot and save training accuracy and loss curves."""
    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], "b-o", label="Training Accuracy")
    if "val_accuracy" in history and history["val_accuracy"]:
        plt.plot(epochs, history["val_accuracy"], "r-s", label="Validation Accuracy")
    plt.title("EfficientNetB0 Breed Classifier - Accuracy Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "training_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], "b-o", label="Training Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(epochs, history["val_loss"], "r-s", label="Validation Loss")
    plt.title("EfficientNetB0 Breed Classifier - Loss Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "training_loss.png", dpi=300)
    plt.close()

    print(f"[SAVED] Training curves saved to {OUTPUTS_DIR}/training_accuracy.png and training_loss.png")

if __name__ == "__main__":
    run_training_pipeline()
