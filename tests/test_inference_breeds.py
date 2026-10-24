import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.prediction import predict_breed

test_dir = Path("dataset/test")
print("=" * 60)
print("     INFERENCE TEST ACROSS ALL 5 CATTLE BREEDS")
print("=" * 60)

for d in sorted(test_dir.iterdir()):
    if d.is_dir():
        sample_img = next(d.glob("*.jpg"))
        res = predict_breed(sample_img)
        print(f"True: {d.name:<12} -> Pred: {res['predicted_breed']:<12} ({res['confidence_percentage']:>6}) | Match: {d.name == res['predicted_breed']}")
        for top in res["top_predictions"]:
            print(f"      - {top['breed']:<12}: {top['confidence'] * 100:.1f}%")
