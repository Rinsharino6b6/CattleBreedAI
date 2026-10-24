# System Architecture - Indian Cattle & Buffalo Breed Recognition System

**SIH Problem Statement ID**: SIH25004  
**Project Phase**: Semester 7 Engineering Project Work Phase 1 Prototype  

---

## 1. System Flowchart Diagram

```text
               +----------------------------------+
               |        User / Examiner           |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   Streamlit Web Interface (app)  |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |     Image Upload & Validation    |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Image Preprocessing (224x224 RGB) |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  EfficientNetB0 Pretrained Base  |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Softmax Classification Head Layer|
               +----------------------------------+
                                |
            +-------------------+-------------------+
            |                                       |
            v                                       v
+-----------------------+               +-----------------------+
|  Top-3 Breed Prob.    |               | Grad-CAM Heatmap XAI  |
+-----------------------+               +-----------------------+
            |                                       |
            v                                       v
+-----------------------+               +-----------------------+
| Confidence Threshold  |               |  Superimposed Overlay |
|  Check (>= 0.60)      |               +-----------------------+
+-----------------------+                           |
            |                                       |
            +-------------------+-------------------+
                                |
                                v
               +----------------------------------+
               |  Breed Metadata & Result Display |
               +----------------------------------+
```

---

## 2. Comprehensive Component Breakdown

### Component 1: Streamlit Interactive Interface (`app.py`)
- Provides a clean multi-page dashboard suited for Phase 1 project demonstrations.
- Navigation sections: Home, Breed Recognition, Model Info, Dataset Info, Evaluation Results, About Project.
- Displays responsive UI cards, progress bars, confidence warnings, and Grad-CAM overlays.

### Component 2: Data Validation & Preprocessing Pipeline (`src/data_utils.py`)
- Standardizes input images to `224 x 224 x 3` resolution.
- Applies color-space conversion (`RGB`) and formats inputs using Keras `preprocess_input`.
- Checks for corrupt images, unsupported extensions, and computes split statistics.

### Component 3: Transfer Learning Backbone (`src/model.py`)
- Uses **EfficientNetB0** pretrained on ImageNet as the core feature extractor.
- Replaces ImageNet head with custom classification head: `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(num_classes, softmax)`.
- Stage 1 trains classification head; Stage 2 unfreezes top 30 convolutional layers for fine-tuning at a learning rate of `1e-5`.

### Component 4: Prediction & Confidence Engine (`src/prediction.py`)
- Generates probability distribution vector across all breeds.
- Sorts probabilities to output top-3 predictions.
- Enforces configurable confidence thresholding (`CONFIDENCE_THRESHOLD = 0.60`) to warn users when inputs are ambiguous or poor quality.

### Component 5: Explainable AI Engine (`src/gradcam.py`)
- Computes gradients of the target predicted class score with respect to the final convolutional feature maps (`top_conv`).
- Performs spatial global average pooling of gradients to compute feature channel weights.
- Generates a 2D intensity heatmap, applies OpenCV `JET` colormap, and overlays it on the original livestock image.

### Component 6: Breed Information Database (`breed_info.json`)
- Stores factual, non-fabricated metadata for Indian cattle breeds (Gir, Sahiwal, Red Sindhi, Kankrej, Tharparkar, Ongole) and buffalo breeds (Murrah, Jaffarabadi, Surti, Mehsana, Nili Ravi).
- Keeps breed info cleanly separated from model prediction logic.
