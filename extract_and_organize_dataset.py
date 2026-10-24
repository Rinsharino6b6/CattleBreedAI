"""
Extract and organize Indian Cattle Breed dataset from Downloads zip archive.
Creates balanced, stratified train/validation/test splits.
"""
import zipfile
import shutil
import random
from pathlib import Path
from PIL import Image

ZIP_PATH = Path(r"C:\Users\ransh\Downloads\Annotated Multiview Image Dataset of Prominent Ind.zip")
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
RAW_DIR = DATASET_DIR / "raw"

RANDOM_SEED = 42
MAX_PER_BREED_ACTIVE = 340  # Balanced count per breed across train/val/test

BREEDS = ["Bargur", "Gir", "Hallikar", "Ongole", "Tharparkar"]

def is_valid_image_bytes(fp) -> bool:
    try:
        with Image.open(fp) as img:
            img.verify()
        return True
    except Exception:
        return False

def main():
    print("=" * 60)
    print("     EXTRACTING & ORGANIZING CATTLE BREED DATASET")
    print("=" * 60)
    print(f"Source Archive : {ZIP_PATH}")
    print(f"Target Dataset : {DATASET_DIR}\n")

    if not ZIP_PATH.exists():
        print(f"[ERROR] Zip archive not found at {ZIP_PATH}")
        return

    random.seed(RANDOM_SEED)

    # Clean existing split directories
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Collect images from zip per breed
    breed_images = {breed: [] for breed in BREEDS}

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for entry in z.namelist():
            parts = entry.strip("/").split("/")
            if len(parts) >= 3 and parts[1] == "newDataset":
                breed = parts[2]
                if breed in breed_images and entry.lower().endswith((".jpg", ".jpeg", ".png")):
                    breed_images[breed].append(entry)

        print("Dataset Census in Zip Archive:")
        for breed, files in breed_images.items():
            print(f"  - {breed:<12}: {len(files):>5} images")

        print("\nExtracting and organizing balanced stratified splits...")

        total_train = 0
        total_val = 0
        total_test = 0

        for breed, files in breed_images.items():
            # Shuffle files deterministically
            random.shuffle(files)

            # Limit active samples to create balanced classes
            active_files = files[:MAX_PER_BREED_ACTIVE]
            n_total = len(active_files)

            n_train = int(n_total * 0.70)
            n_val = int(n_total * 0.15)
            n_test = n_total - n_train - n_val

            train_files = active_files[:n_train]
            val_files = active_files[n_train:n_train + n_val]
            test_files = active_files[n_train + n_val:]

            for split_name, split_dir, split_files in [
                ("train", TRAIN_DIR, train_files),
                ("val", VAL_DIR, val_files),
                ("test", TEST_DIR, test_files),
            ]:
                breed_split_dir = split_dir / breed
                breed_split_dir.mkdir(parents=True, exist_ok=True)

                for fpath in split_files:
                    fname = Path(fpath).name
                    dest_path = breed_split_dir / fname
                    with z.open(fpath) as src_file, open(dest_path, "wb") as dst_file:
                        shutil.copyfileobj(src_file, dst_file)

            total_train += len(train_files)
            total_val += len(val_files)
            total_test += len(test_files)

            print(f"  Processed '{breed:<10}': Train={len(train_files):>3} | Val={len(val_files):>2} | Test={len(test_files):>2} (Total={n_total})")

        print("-" * 60)
        print(f"Summary:")
        print(f"  Training Images   : {total_train}")
        print(f"  Validation Images : {total_val}")
        print(f"  Test Images       : {total_test}")
        print(f"  Grand Total       : {total_train + total_val + total_test}")
        print("=" * 60)
        print("[SUCCESS] Dataset extraction and stratified partitioning complete!")

if __name__ == "__main__":
    main()
