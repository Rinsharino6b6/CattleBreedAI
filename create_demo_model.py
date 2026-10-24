import json
import sys
from pathlib import Path

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import (
    MODEL_PATH, CLASS_NAMES_PATH, DEFAULT_CLASS_NAMES, IMAGE_SIZE
)
from src.model import build_transfer_model

def initialize_demo_model():
    """Build and save the EfficientNetB0 model checkpoint for instant analysis."""
    print("Initializing EfficientNetB0 Model Checkpoint...")

    num_classes = len(DEFAULT_CLASS_NAMES)
    model = build_transfer_model(
        num_classes=num_classes,
        base_architecture="EfficientNetB0",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Ensure models directory exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"[SUCCESS] Saved model checkpoint to: {MODEL_PATH}")

    # Save class names JSON
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(DEFAULT_CLASS_NAMES, f, indent=2)
    print(f"[SUCCESS] Saved class names to: {CLASS_NAMES_PATH}\n")

if __name__ == "__main__":
    initialize_demo_model()
