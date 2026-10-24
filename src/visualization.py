import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

def plot_confusion_matrix(
    cm_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Generate and save a clean, publication-ready confusion matrix.
    Labels X-axis: Predicted Breed, Y-axis: Actual Breed.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )

    ax.set_title("Indian Cattle & Buffalo Breed Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Breed", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Breed", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if not show_plot:
        plt.close(fig)

    return fig

def plot_top_predictions_bar(
    predictions: List[Dict[str, Any]],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Generate horizontal bar chart for top-k prediction probabilities."""
    breeds = [p["breed"] for p in predictions]
    confidences = [p["confidence"] * 100 for p in predictions]

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#2E7D32", "#4CAF50", "#81C784"][:len(breeds)]

    bars = ax.barh(breeds, confidences, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Top Predictions", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.close(fig)
    return fig
