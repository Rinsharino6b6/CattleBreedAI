import os
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from src.config import (
    DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, DEFAULT_CLASS_NAMES, RANDOM_SEED
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def is_valid_image(file_path: Path) -> bool:
    """Check if an image file exists, has a valid extension, and can be opened."""
    if not file_path.is_file():
        return False
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def analyze_image_quality(image_input: Any) -> Dict[str, Any]:
    """
    Compute image quality parameters: resolution, aspect ratio, brightness,
    contrast score, and blurriness index using OpenCV.
    """
    if isinstance(image_input, (str, Path)):
        img_pil = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img_pil = image_input
    else:
        img_pil = Image.fromarray(image_input)

    width, height = img_pil.size
    aspect_ratio = round(width / height, 2)

    # Convert PIL to cv2 BGR / Gray for metric calculation
    np_img = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    brightness = round(float(np.mean(gray)), 1)
    contrast = round(float(np.std(gray)), 1)
    
    # Laplacian variance for focus / blurriness score
    blur_score = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)

    is_sharp = blur_score > 100.0
    is_good_lighting = 40.0 <= brightness <= 220.0

    quality_status = "Good Quality"
    if not is_sharp:
        quality_status = "Slightly Blurry"
    elif brightness < 40:
        quality_status = "Dark Lighting"
    elif brightness > 220:
        quality_status = "Overexposed"

    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "quality_status": quality_status,
        "is_sharp": is_sharp,
        "is_good_lighting": is_good_lighting
    }

def scan_dataset_structure(dataset_root: Path = DATASET_DIR) -> Dict[str, Any]:
    """
    Scan dataset directory and calculate class counts, splits,
    corrupted image detection, and class balance analysis.
    """
    summary = {
        "exists": dataset_root.exists(),
        "splits": {},
        "class_counts": {},
        "total_valid_images": 0,
        "corrupted_files": [],
        "classes": [],
        "imbalance_warning": False
    }

    if not dataset_root.exists():
        return summary

    splits_to_check = ["train", "validation", "test"]
    has_split_folders = any((dataset_root / s).exists() for s in splits_to_check)

    if has_split_folders:
        for split in splits_to_check:
            split_dir = dataset_root / split
            if not split_dir.exists():
                continue
            summary["splits"][split] = {}
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    class_name = class_dir.name
                    valid_count = 0
                    for img_file in class_dir.iterdir():
                        if img_file.is_file():
                            if is_valid_image(img_file):
                                valid_count += 1
                            else:
                                summary["corrupted_files"].append(str(img_file))
                    summary["splits"][split][class_name] = valid_count
                    summary["class_counts"][class_name] = summary["class_counts"].get(class_name, 0) + valid_count
                    if class_name not in summary["classes"]:
                        summary["classes"].append(class_name)
    else:
        for class_dir in sorted(dataset_root.iterdir()):
            if class_dir.is_dir() and class_dir.name not in ["models", "outputs", "reports"]:
                class_name = class_dir.name
                valid_count = 0
                for img_file in class_dir.iterdir():
                    if img_file.is_file():
                        if is_valid_image(img_file):
                            valid_count += 1
                        else:
                            summary["corrupted_files"].append(str(img_file))
                summary["class_counts"][class_name] = valid_count
                summary["classes"].append(class_name)

    summary["total_valid_images"] = sum(summary["class_counts"].values())
    summary["classes"] = sorted(summary["classes"])

    if summary["class_counts"]:
        counts = [c for c in summary["class_counts"].values() if c > 0]
        if counts:
            min_c, max_c = min(counts), max(counts)
            if min_c > 0 and (max_c / min_c) > 3.0:
                summary["imbalance_warning"] = True

    return summary

def get_dataset_classes(dataset_root: Path = DATASET_DIR) -> List[str]:
    """Dynamically get class names from dataset directory or return defaults."""
    summary = scan_dataset_structure(dataset_root)
    if summary["classes"]:
        return summary["classes"]
    return DEFAULT_CLASS_NAMES

def load_and_preprocess_image(image_path_or_bytes: Any, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load, resize, and preprocess an image tensor (1, 224, 224, 3) for EfficientNetB0."""
    if isinstance(image_path_or_bytes, (str, Path)):
        img = Image.open(image_path_or_bytes)
    elif isinstance(image_path_or_bytes, Image.Image):
        img = image_path_or_bytes
    else:
        img = Image.open(image_path_or_bytes)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)

    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def create_data_generators(
    dataset_dir: Path = DATASET_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE
) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], List[str]]:
    """Create training, validation, and test dataset generators."""
    train_path = dataset_dir / "train"
    val_path = dataset_dir / "validation"
    test_path = dataset_dir / "test"

    if not train_path.exists():
        return None, None, None, get_dataset_classes(dataset_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED
    )

    class_names = train_ds.class_names

    val_ds = None
    if val_path.exists():
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_path,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False
        )

    test_ds = None
    if test_path.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_path,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False
        )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    if val_ds:
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    if test_ds:
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
