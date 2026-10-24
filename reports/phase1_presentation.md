# Semester 7 Phase 1 Project Review Presentation

**Title**: Image Based Breed Recognition for Cattle and Buffaloes of India  
**SIH Problem Statement ID**: SIH25004 | **Theme**: Agriculture, FoodTech & Rural Development  

---

## Slide 1: Title Slide
- **Project Name**: AI-Powered Indian Cattle & Buffalo Breed Recognition System
- **Problem Statement ID**: SIH25004
- **Domain**: Computer Vision & Smart Agriculture
- **Presented By**: Final Year Engineering Project Group
- **Guide Name**: [Project Guide / Mentor Name]

---

## Slide 2: SIH Problem Statement
- **Title**: Image Based Breed Recognition for Cattle and Buffaloes of India
- **Objective**: Build an automated computer vision application to identify indigenous Indian cattle and buffalo breeds from photographs.
- **Key Target Users**: Farmers, livestock inspectors, insurance agencies, and veterinary authorities.

---

## Slide 3: Motivation & Importance
- India possesses rich livestock biodiversity (over 50 registered indigenous cattle breeds and 19 buffalo breeds).
- Indigenous breeds (Gir, Sahiwal, Murrah) display superior heat tolerance and disease resistance.
- Manual breed identification requires domain expertise and is prone to misclassification.

---

## Slide 4: Existing System
- Visual inspection by domain experts based on physical morphological features.
- Manual record-keeping and physical ear tagging/RFID tags.
- Simple color or size estimation without automated feature extraction.

---

## Slide 5: Limitations of Existing Systems
- Subjective human bias and scarcity of trained livestock experts in rural areas.
- Physical tags can fall off, be tampered with, or cause infection.
- Existing software solutions lack explainable AI and confidence thresholding.

---

## Slide 6: Proposed Solution
- **Deep Transfer Learning System**: EfficientNetB0 backbone fine-tuned on Indian cattle and buffalo breeds.
- **Explainable AI**: Grad-CAM heatmap visualization to justify model predictions.
- **Interactive UI**: Multi-page Streamlit web dashboard for real-time inference.

---

## Slide 7: Project Objectives (Phase 1)
1. Implement dynamic multiclass classification supporting 10+ major Indian breeds.
2. Develop dataset validation (`validate_dataset.py`) and automated 70/15/15 stratified dataset splitter (`prepare_dataset.py`).
3. Compute standardized evaluation metrics (Accuracy, Precision, Recall, Macro F1, Confusion Matrix).
4. Build a user-friendly Streamlit prototype interface.

---

## Slide 8: System Architecture
- Flow diagram showing: Image Input → Preprocessing (224x224) → EfficientNetB0 Extractor → Softmax Dense Head → Top-3 Probabilities & Grad-CAM Attention Heatmap.

---

## Slide 9: Dataset Structure & Initial Breed Classes
- **Cattle Breeds**: Gir, Sahiwal, Red Sindhi, Kankrej, Tharparkar, Ongole.
- **Buffalo Breeds**: Murrah, Jaffarabadi, Surti, Mehsana, Nili Ravi.
- **Split Strategy**: 70% Training, 15% Validation, 15% Test.

---

## Slide 10: Methodology & Data Augmentation
- Resizing & Normalization to 224x224x3.
- Augmentation: Horizontal flips, small rotations (±10°), zoom (±10%), translation, and contrast adjustments to preserve anatomical features.

---

## Slide 11: Deep Learning Model Architecture
- **Base Model**: EfficientNetB0 pretrained on ImageNet (~5.3M parameters).
- **Custom Classification Head**: GlobalAveragePooling2D → Dropout(0.3) → Dense(Softmax).
- **Fallback Support**: MobileNetV2 for lightweight/edge deployment.

---

## Slide 12: Two-Stage Training Pipeline
- **Stage 1 (Head Training)**: Frozen base model, Adam optimizer (lr=0.001), 10 epochs.
- **Stage 2 (Fine-Tuning)**: Unfreeze top 30 conv layers, low learning rate (lr=0.00001), 10 epochs.
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.

---

## Slide 13: Explainable AI with Grad-CAM
- Explains *why* the model predicted a specific breed.
- Highlights model attention on key regions (horns, forehead curvature, hump, ears, coat color).
- Provides visual trust for non-expert end-users.

---

## Slide 14: Experimental Results & Demonstration
- Test set performance metrics (Accuracy, Macro F1, Confusion Matrix).
- Working prototype demonstration via Streamlit dashboard.
- *[Placeholder for live screenshot / demo video clip]*

---

## Slide 15: Limitations & Semester 8 Future Scope
- **Current Limitations**: Requires good lighting; relies on clear side/head views of livestock.
- **Semester 8 Future Scope**:
  - React/Node.js full-stack integration with MongoDB.
  - Mobile App development (React Native / Flutter).
  - Stage-1 Cattle vs Buffalo detector & edge deployment on Raspberry Pi.
