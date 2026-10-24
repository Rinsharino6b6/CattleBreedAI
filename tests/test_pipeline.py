import unittest
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_CLASS_NAMES, get_animal_type, IMAGE_SIZE
from src.data_utils import is_valid_image, load_and_preprocess_image
from src.model import build_transfer_model

class TestCattleBreedRecognitionPipeline(unittest.TestCase):

    def test_01_config_and_breed_mapping(self):
        """Test configuration defaults and animal category mapping."""
        self.assertGreater(len(DEFAULT_CLASS_NAMES), 5)
        self.assertEqual(get_animal_type("Gir"), "Cattle")
        self.assertEqual(get_animal_type("Murrah"), "Buffalo")

    def test_02_image_preprocessing(self):
        """Test creating synthetic test image and pre-processing pipeline."""
        test_img_path = PROJECT_ROOT / "tests" / "temp_synthetic_cow.jpg"
        test_img_path.parent.mkdir(parents=True, exist_ok=True)

        # Create synthetic RGB image
        synthetic_array = np.uint8(np.random.rand(300, 300, 3) * 255)
        img = Image.fromarray(synthetic_array)
        img.save(test_img_path)

        self.assertTrue(is_valid_image(test_img_path))

        processed_tensor = load_and_preprocess_image(test_img_path)
        self.assertEqual(processed_tensor.shape, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

        # Cleanup temp image
        if test_img_path.exists():
            test_img_path.unlink()

    def test_03_model_builder(self):
        """Test constructing EfficientNetB0 transfer learning model architecture."""
        model = build_transfer_model(num_classes=5, base_architecture="EfficientNetB0")
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape, (None, 5))

if __name__ == "__main__":
    unittest.main()
