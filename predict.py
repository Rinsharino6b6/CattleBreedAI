import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.prediction import predict_breed
from src.config import MODEL_PATH

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py dataset/test/Gir/cow_01.jpg\n")
        
        if not MODEL_PATH.exists():
            print(f"[STATUS] Trained model missing at: {MODEL_PATH}")
            print("Please train model using: python train.py")
        else:
            print(f"[STATUS] Trained model is ready at: {MODEL_PATH}")
        sys.exit(0)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"[ERROR] Image file does not exist: {image_path}")
        sys.exit(1)

    try:
        res = predict_breed(image_path)

        print("=" * 50)
        print("       INDIAN LIVESTOCK BREED PREDICTION RESULT")
        print("=" * 50)
        print(f" Animal Type     : {res['animal_type']}")
        print(f" Predicted Breed : {res['predicted_breed']}")
        print(f" Confidence      : {res['confidence_percentage']}")
        print("=" * 50)

        print("\nTop 3 Predictions:")
        print("-" * 35)
        for i, pred in enumerate(res["top_predictions"], 1):
            print(f" {i}. {pred['breed']:<15} : {pred['confidence'] * 100:>5.1f}%")
        print("-" * 35)

        if not res["is_confident"]:
            print(f"\n[WARNING] {res['warning_message']}")

    except Exception as e:
        print(f"[ERROR] Failed to run prediction pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
