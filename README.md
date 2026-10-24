# CattleBreedAI - Intelligent Livestock Breed Recognition & Veterinary Healthcare Platform

**CattleBreedAI** is an AI-powered computer vision and veterinary healthcare application designed for indigenous Indian cattle (*Bos indicus*) and buffalo (*Bubalus bubalis*) breeds.

The platform provides dual capabilities:
1. **Computer Vision & Explainable AI**: Automatic breed recognition using deep transfer learning (EfficientNetB0) paired with Grad-CAM visual attention heatmaps.
2. **Clinical Disease Assistance & Veterinary Support**: Symptom checking, emergency triaging, verified government veterinary helplines (including **1962** Mobile Veterinary Emergency Ambulance), and an integrated consultation ticket management system.

---

## 🛠️ Technology Stack
- **Language**: Python 3.10 - 3.13
- **Deep Learning Framework**: TensorFlow 2.x / Keras
- **Core Vision Architecture**: EfficientNetB0 (ImageNet Transfer Learning, MobileNetV2 fallback)
- **Explainable AI**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Image Processing**: OpenCV, Pillow (PIL)
- **Clinical Database & Ticketing**: SQLite3, JSON clinical knowledge base
- **Data Analytics & Visualization**: NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
- **User Interface**: Streamlit Web Dashboard

---

## 📁 Project Directory Structure
```text
CattleBreedAI/
│
├── data/                               # Clinical & veterinary directories
│   ├── cattle_diseases.json            # 10 verified cattle/buffalo disease profiles
│   ├── veterinary_directory.json       # Genuine national & state veterinary helplines
│   └── veterinary_support.db           # Persistent SQLite consultation ticket database
│
├── dataset/                            # Dataset directory (train / validation / test)
│   ├── train/                          # Training split per breed
│   ├── validation/                     # Validation split per breed
│   └── test/                           # Test split per breed
│
├── models/                             # Model artifacts
│   ├── breed_classifier.keras          # Saved model weights checkpoint
│   ├── class_names.json                # Class mapping JSON
│   └── training_history.json           # Epoch metrics history
│
├── outputs/                            # Visualizations & consultation uploads
│   ├── dataset_class_distribution.png
│   ├── confusion_matrix.png
│   ├── consultation_uploads/           # Photos attached to consultation tickets
│   └── gradcam/                        # Grad-CAM heatmap visualizations
│
├── reports/                            # Technical reports & architecture diagrams
│   ├── system_architecture.md
│   └── classification_report.txt
│
├── src/                                # Modular source code
│   ├── __init__.py
│   ├── config.py                       # Configuration & paths
│   ├── data_utils.py                   # Preprocessing & image quality checks
│   ├── model.py                        # EfficientNetB0 transfer learning architecture
│   ├── prediction.py                   # Inference engine & thresholding
│   ├── gradcam.py                      # Grad-CAM XAI heatmap generator
│   ├── disease_assistant.py            # Clinical symptom checker & emergency triager
│   ├── veterinary_support.py           # SQLite ticket manager & directory queries
│   └── visualization.py                # Metrics & confusion matrix plotters
│
├── tests/                              # Automated test suites
│   ├── test_pipeline.py
│   └── test_disease_assistant.py
│
├── app.py                              # Main interactive Streamlit application
├── train.py                            # Standalone model training script
├── evaluate.py                         # Standalone test evaluation script
├── predict.py                          # CLI prediction utility
├── prepare_dataset.py                  # Stratified 70/15/15 data splitter
├── validate_dataset.py                 # Dataset health & balance validator
├── breed_info.json                     # Factual metadata for 11 Indian breeds
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation
```

---

## 🚀 Key System Features

### 1. Breed Recognition & Grad-CAM Explainable AI
- Classifies 11 major Indian breeds: Gir, Sahiwal, Red Sindhi, Kankrej, Tharparkar, Ongole (Cattle); Murrah, Jaffarabadi, Surti, Mehsana, Nili Ravi (Buffalo).
- Computes Top-3 probability distribution and flags uncertain / non-livestock images.
- Grad-CAM heatmap highlights spatial focus regions (convex forehead, horn geometry, dewlap folds, ears).

### 2. Cattle Disease Assistance & Symptom Checker
- Intuitive checkbox symptom picker grouped by anatomical areas (Mouth, Breathing, Skin/Legs, Udder, Digestion, Mobility).
- Matches symptoms against 10 major conditions: Foot & Mouth Disease (FMD), Lumpy Skin Disease (LSD), Mastitis, Hemorrhagic Septicemia (HS), Black Quarter (BQ), Acute Bloat (Afara), Milk Fever, Brucellosis, Theileriosis, and Bovine Respiratory Disease.
- Provides immediate non-prescription first aid, hygiene steps, and vaccination guidance.

### 3. Automated Emergency Triaging & 1962 Speed-Dial
- Real-time rule engine detects critical emergencies:
  - Acute Bloat / Afara (Asphyxiation risk)
  - Hemorrhagic Septicemia throat edema & loud grunting
  - Milk Fever sternal recumbency & hypothermia
  - Black Quarter crepitating muscle swelling
  - Red Water (Babesiosis) hemolysis
- Displays a prominent red emergency banner with direct instructions to dial **1962** (National Mobile Veterinary Emergency Ambulance).

### 4. Veterinary Consultation Ticket System
- Farmers can submit cases with animal details, descriptions, and optional photos.
- Generates a verifiable Ticket ID (e.g., `VET-2026-X8F4K`) persisted in a local SQLite database.
- Includes a Ticket Tracking tab to check status.

### 5. Verified Veterinary Directory & Vaccination Calendar
- Genuine national toll-free helplines: **1962** (MVU Ambulance), **1800-180-1551** (Kisan Call Center), **011-23382753** (DAHD).
- State-wise searchable contacts for animal husbandry directorates and apex veterinary teaching hospitals across 12 states.
- National livestock vaccination timetable.

---

## ⚙️ Installation & Execution

### 1. Setup Virtual Environment
```powershell
cd C:\projects\CattleBreedAI
py -3.13 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```powershell
python -m unittest tests/test_disease_assistant.py
```

### 3. Launch the Application
```powershell
streamlit run app.py
```
Open browser at `http://localhost:8501`.

---

## ⚠️ Veterinary & Medical Disclaimer
This software serves as an agricultural decision-support and educational tool. It **does NOT prescribe prescription antibiotics or prescription drugs**, nor does it replace in-person clinical examinations or laboratory testing by a registered veterinary practitioner. In any acute or life-threatening situation, contact local veterinary authorities or dial **1962** immediately.
