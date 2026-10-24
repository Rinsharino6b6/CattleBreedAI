# Semester 7 Project Work Phase 1 - Project Report Outline

**Project Title**: Image Based Breed Recognition for Cattle and Buffaloes of India  
**SIH Problem Statement ID**: SIH25004  
**Category**: Software  
**Theme**: Agriculture, FoodTech & Rural Development  

---

## Table of Contents

### 1. Introduction
- Background of Indian livestock economy (cattle and buffalo breeds).
- Importance of indigenous breed conservation and dairy farming productivity.
- Role of computer vision in agricultural technology.

### 2. Problem Statement
- SIH25004 Problem Statement definition.
- Challenges in manual visual identification of Indian zebu cattle and river buffaloes.

### 3. Motivation
- Need for automated, low-cost, accurate breed recognition for farmers, insurance assessors, and veterinary officers.

### 4. Existing System
- Manual physical inspection by experts based on ear shape, horn curvature, and dewlap folds.
- Basic RFID/ear tagging systems without automated visual verification.

### 5. Limitations of Existing Systems
- Subjective human errors and dependency on domain experts.
- High cost and inconvenience of physical tagging.
- Non-scalable for rural pan-India adoption.

### 6. Proposed System
- AI-based computer vision solution using Transfer Learning (EfficientNetB0) and Grad-CAM Explainable AI integrated into a Streamlit dashboard.

### 7. Objectives
- Develop an end-to-end multiclass livestock breed classifier.
- Implement Grad-CAM feature attention visualization.
- Establish dataset validation and 70/15/15 stratified data splitting pipelines.

### 8. Literature Survey
- *[Reference 1 Placeholder]*: Survey on CNN architectures in livestock classification.
- *[Reference 2 Placeholder]*: Deep transfer learning for Zebu cattle breed identification.
- *[Reference 3 Placeholder]*: Explainable AI in agricultural computer vision applications.

### 9. Methodology
- Overview of system design from dataset curation to web interface deployment.

### 10. System Architecture
- Functional block diagrams, module components, and data flow pipelines.

### 11. Dataset
- Class structure for 11 initial Indian breeds (Gir, Sahiwal, Red Sindhi, Kankrej, Tharparkar, Ongole, Murrah, Jaffarabadi, Surti, Mehsana, Nili Ravi).

### 12. Data Preprocessing
- Image format validation, resizing to 224x224, color-space normalization, and corrupted file removal.

### 13. Data Augmentation
- Invariance transformations: Random horizontal flip, small rotation (±10°), zoom (±10%), translation, and contrast adjustments.

### 14. Model Selection
- Comparative analysis of EfficientNetB0 vs ResNet50 and MobileNetV2.

### 15. Transfer Learning
- Rationale for leveraging ImageNet pretrained weights and fine-tuning upper convolutional layers.

### 16. Training Process
- Two-stage training procedure: Stage 1 frozen base head training (lr=0.001); Stage 2 fine-tuning top layers (lr=0.00001).

### 17. Evaluation Metrics
- Formal mathematical definitions of Accuracy, Precision, Recall, Macro F1, and Weighted F1 scores.

### 18. Results
- *[Results Placeholder]*: Test accuracy, loss curves, and summary table (populated after training on real dataset).

### 19. Confusion Matrix
- *[Confusion Matrix Placeholder]*: Analysis of actual vs predicted breed classifications.

### 20. Grad-CAM Explainable AI
- Interpretation of gradient heatmaps showing model attention on horns, forehead, and dewlap regions.

### 21. Current Implementation Status
- Working prototype features implemented in Phase 1.

### 22. Limitations
- Dependence on image quality, lighting variations, occlusions, and dataset class balance.

### 23. Future Scope (Semester 8 Plan)
- Mobile application development, cloud API backend, IoT edge integration, and separate Cattle/Buffalo stage-1 detector.

### 24. Conclusion
- Summary of Phase 1 prototype achievements and readiness for Phase 2.

### 25. References
- *[Insert academic papers, SIH problem statement link, and TensorFlow documentation links here]*
