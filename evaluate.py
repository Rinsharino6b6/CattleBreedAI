import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

sys.path.append(str(Path(__file__).resolve().parent))

from src.config import (
    MODEL_PATH, CLASS_NAMES_PATH, TEST_DIR, OUTPUTS_DIR, REPORTS_DIR, IMAGE_SIZE, BATCH_SIZE
)
from src.visualization import plot_confusion_matrix

def evaluate_model_on_test_set():
    """Evaluate trained model strictly on TEST dataset and output metrics, confusion matrix, & report."""
    print("=" * 60)
    print("         INDIAN CATTLE & BUFFALO MODEL EVALUATION")
    print("=" * 60)

    # Check model file exists
    if not MODEL_PATH.exists():
        print(f"[ERROR] Trained model file not found at: {MODEL_PATH}")
        print("Please train the model first by running: python train.py\n")
        return

    # Check test dataset exists
    if not TEST_DIR.exists() or not any(TEST_DIR.iterdir()):
        print(f"[ERROR] Test dataset folder is missing or empty at: {TEST_DIR}")
        print("Please place test split data in dataset/test/ before evaluation.\n")
        return

    # Load class names
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
    else:
        class_names = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])

    print(f"Loading trained model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print(f"Loading test images from: {TEST_DIR}")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    y_true = []
    y_pred_probs = []

    print("\nRunning inference on test dataset...")
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred_probs.append(probs)
        y_true.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print("              OVERALL TEST METRICS")
    print("=" * 50)
    print(f" Accuracy          : {accuracy * 100:.2f}%")
    print(f" Macro Precision   : {precision_macro * 100:.2f}%")
    print(f" Macro Recall      : {recall_macro * 100:.2f}%")
    print(f" Macro F1-Score    : {f1_macro * 100:.2f}%")
    print(f" Weighted F1-Score : {f1_weighted * 100:.2f}%")
    print("=" * 50)

    # Generate detailed text classification report
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file_path = REPORTS_DIR / "classification_report.txt"
    with open(report_file_path, "w") as f:
        f.write("=====================================================\n")
        f.write(" INDIAN CATTLE & BUFFALO BREED RECOGNITION REPORT   \n")
        f.write("=====================================================\n\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Macro F1-Score: {f1_macro * 100:.2f}%\n")
        f.write(f"Weighted F1-Score: {f1_weighted * 100:.2f}%\n\n")
        f.write("Detailed Per-Class Metrics:\n")
        f.write("-----------------------------------------------------\n")
        f.write(report_str)

    print(f"\n[SAVED] Classification report saved to: {report_file_path}")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, save_path=cm_path)
    print(f"[SAVED] Confusion matrix plot saved to: {cm_path}\n")

if __name__ == "__main__":
    evaluate_model_on_test_set()
