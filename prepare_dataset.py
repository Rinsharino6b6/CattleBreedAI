import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))

from src.config import DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, RANDOM_SEED
from src.data_utils import ALLOWED_EXTENSIONS, is_valid_image

def split_and_organize_dataset(
    source_dir: Path,
    target_dir: Path = DATASET_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED
) -> None:
    """
    Split a raw dataset folder containing class directories into train, val, and test sets.
    Uses stratified splitting to maintain class proportions without data leakage.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    source_dir = Path(source_dir).resolve()

    if not source_dir.exists():
        print(f"[ERROR] Source directory does not exist: {source_dir}")
        return

    # Check for class directories in source
    class_folders = [d for d in source_dir.iterdir() if d.is_dir() and d.name not in ["train", "validation", "test", "models", "outputs", "reports"]]
    if not class_folders:
        print(f"[ERROR] No class directories found inside {source_dir}")
        return

    print(f"Preparing to split dataset from: {source_dir}")
    print(f"Split proportions: Train={train_ratio:.0%}, Validation={val_ratio:.0%}, Test={test_ratio:.0%}\n")

    # Create target split directories
    for split_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        split_path.mkdir(parents=True, exist_ok=True)

    total_moved = 0

    for class_dir in class_folders:
        class_name = class_dir.name
        valid_images = [f for f in class_dir.iterdir() if is_valid_image(f)]

        if len(valid_images) == 0:
            print(f"[SKIP] Class '{class_name}' has 0 valid images.")
            continue

        # Create class subfolders in train, val, test
        (TRAIN_DIR / class_name).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / class_name).mkdir(parents=True, exist_ok=True)
        (TEST_DIR / class_name).mkdir(parents=True, exist_ok=True)

        if len(valid_images) < 3:
            # Too few images to split stratified, copy all to train
            print(f"[NOTE] Class '{class_name}' has too few images ({len(valid_images)}). Copying all to train.")
            for img in valid_images:
                shutil.copy2(img, TRAIN_DIR / class_name / img.name)
            total_moved += len(valid_images)
            continue

        # Split 1: Split into train and temp (val + test)
        temp_ratio = val_ratio + test_ratio
        train_files, temp_files = train_test_split(
            valid_images,
            test_size=temp_ratio,
            random_state=seed,
            shuffle=True
        )

        # Split 2: Split temp into val and test
        val_relative_ratio = val_ratio / temp_ratio
        if len(temp_files) >= 2:
            val_files, test_files = train_test_split(
                temp_files,
                test_size=(1.0 - val_relative_ratio),
                random_state=seed,
                shuffle=True
            )
        else:
            val_files = temp_files
            test_files = []

        # Copy files to respective split destinations
        for f in train_files:
            shutil.copy2(f, TRAIN_DIR / class_name / f.name)
        for f in val_files:
            shutil.copy2(f, VAL_DIR / class_name / f.name)
        for f in test_files:
            shutil.copy2(f, TEST_DIR / class_name / f.name)

        total_moved += len(valid_images)
        print(f"  Class '{class_name:<15}': Total={len(valid_images):>4} | Train={len(train_files):>4} | Val={len(val_files):>3} | Test={len(test_files):>3}")

    print(f"\n[SUCCESS] Successfully organized {total_moved} images across train, validation, and test splits!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    else:
        src = DATASET_DIR
    split_and_organize_dataset(src)
