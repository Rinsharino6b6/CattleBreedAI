import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "breed_classifier.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.json"

OUTPUTS_DIR = BASE_DIR / "outputs"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
CONSULTATION_UPLOADS_DIR = OUTPUTS_DIR / "consultation_uploads"
REPORTS_DIR = BASE_DIR / "reports"

BREED_INFO_PATH = BASE_DIR / "breed_info.json"

# Veterinary Healthcare & Disease Assistance Data Paths
DATA_DIR = BASE_DIR / "data"
DISEASES_INFO_PATH = DATA_DIR / "cattle_diseases.json"
VET_DIRECTORY_PATH = DATA_DIR / "veterinary_directory.json"
VET_DB_PATH = DATA_DIR / "veterinary_support.db"

# Create required directories
for folder in [DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, GRADCAM_DIR, CONSULTATION_UPLOADS_DIR, REPORTS_DIR, DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Application & Model Metadata
APP_TITLE = "CattleBreedAI"
APP_SUBTITLE = "Intelligent Cattle & Buffalo Breed Recognition and Veterinary Healthcare Platform"
PROJECT_NAME = "CattleBreedAI - Indian Livestock Recognition & Healthcare System"

# Image & Model Parameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 8
INITIAL_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.00001
RANDOM_SEED = 42

# Confidence Threshold
CONFIDENCE_THRESHOLD = 0.60

# Configured Breed Classes
DEFAULT_CLASS_NAMES = [
    "Bargur",
    "Gir",
    "Hallikar",
    "Ongole",
    "Tharparkar",
    "Sahiwal",
    "Red_Sindhi",
    "Kankrej",
    "Murrah",
    "Jaffarabadi",
    "Surti",
    "Mehsana",
    "Nili_Ravi"
]

# Breed Category Mapping
ANIMAL_TYPE_MAP = {
    "Bargur": "Cattle",
    "Gir": "Cattle",
    "Hallikar": "Cattle",
    "Ongole": "Cattle",
    "Tharparkar": "Cattle",
    "Sahiwal": "Cattle",
    "Red_Sindhi": "Cattle",
    "Red Sindhi": "Cattle",
    "Kankrej": "Cattle",
    "Murrah": "Buffalo",
    "Jaffarabadi": "Buffalo",
    "Surti": "Buffalo",
    "Mehsana": "Buffalo",
    "Nili_Ravi": "Buffalo",
    "Nili Ravi": "Buffalo"
}

def get_animal_type(breed_name: str) -> str:
    """Resolve animal category (Cattle vs Buffalo) for a given breed."""
    normalized_name = breed_name.replace(" ", "_")
    return ANIMAL_TYPE_MAP.get(breed_name, ANIMAL_TYPE_MAP.get(normalized_name, "Livestock"))
