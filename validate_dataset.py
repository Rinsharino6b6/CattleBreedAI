import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import DATASET_DIR, OUTPUTS_DIR
from src.data_utils import scan_dataset_structure

def validate_and_report_dataset(dataset_path: Path = DATASET_DIR) -> None:
    """Scan dataset, print summary report, warn about imbalances or corruptions, and plot distribution."""
    print("=" * 60)
    print("      INDIAN CATTLE & BUFFALO DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Scanning directory: {dataset_path.resolve()}\n")

    summary = scan_dataset_structure(dataset_path)

    if not summary["exists"] or summary["total_valid_images"] == 0:
        print("[WARNING] Dataset directory is empty or missing!")
        print(f"Please place class folders inside: {dataset_path}")
        print("Example structure:")
        print("  dataset/train/Gir/")
        print("  dataset/train/Sahiwal/")
        print("  dataset/train/Murrah/\n")
        return

    print("Dataset Summary:")
    print("-" * 40)
    for class_name, count in summary["class_counts"].items():
        print(f"  {class_name:<20}: {count:>5} images")

    print("-" * 40)
    print(f"Total valid images    : {summary['total_valid_images']}")
    print(f"Number of classes     : {len(summary['classes'])}")

    if summary["splits"]:
        print("\nSplit Counts:")
        for split_name, split_data in summary["splits"].items():
            split_total = sum(split_data.values())
            print(f"  - {split_name.capitalize():<12}: {split_total:>5} images")

    if summary["corrupted_files"]:
        print(f"\n[WARNING] Found {len(summary['corrupted_files'])} corrupted/invalid files:")
        for corrupt in summary["corrupted_files"][:5]:
            print(f"  - {corrupt}")
        if len(summary["corrupted_files"]) > 5:
            print(f"  ... and {len(summary['corrupted_files']) - 5} more.")

    # Class imbalance check
    if summary["imbalance_warning"]:
        print("\n[WARNING] Severe class imbalance detected! Some classes have >3x more samples than others.")
        print("Consider collecting more samples or using class weighting during training.")

    # Low sample count check
    low_count_classes = [c for c, count in summary["class_counts"].items() if count < 20]
    if low_count_classes:
        print(f"\n[WARNING] Classes with fewer than 20 images: {', '.join(low_count_classes)}")
        print("Transfer learning may overfit on these classes without sufficient samples.")

    # Plot class distribution chart
    plt.figure(figsize=(10, 5))
    classes = list(summary["class_counts"].keys())
    counts = list(summary["class_counts"].values())

    sns.barplot(x=classes, y=counts, palette="viridis")
    plt.title("Indian Livestock Breed Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Breed Class", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    chart_path = OUTPUTS_DIR / "dataset_class_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n[SUCCESS] Class distribution chart saved to: {chart_path}\n")

if __name__ == "__main__":
    validate_and_report_dataset()
