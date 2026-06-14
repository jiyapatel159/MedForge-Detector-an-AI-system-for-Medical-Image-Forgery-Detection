# 🔬 MedForge Detector — Medical Image Forgery Detection

An AI-powered system that detects forgery in medical images (X-rays, MRI scans, CT scans) using a Convolutional Neural Network built completely from scratch — no pretrained weights. The system also provides Grad-CAM heatmaps to visually explain *where* tampering was detected, and supports DICOM file uploads with metadata inspection.

---

## 📌 Overview

Medical images can be digitally tampered with — tumors removed, regions copy-pasted, or false annotations added — leading to misdiagnosis, insurance fraud, or false legal evidence. MedForge Detector automatically classifies medical images as **Real** or **Forged** and highlights the suspicious region using Grad-CAM.

**Detected forgery types:**
- Copy-move forgery
- Copy-paste (splicing)
- Content removal (inpainting)
- Text addition

---

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | **93.04%** |
| Validation Accuracy | 94.21% |
| Precision | 94.42% |
| Recall | 94.88% |
| F1 Score | 94.65% |
| AUC-ROC | 0.9809 |

Model: 5-block CNN trained from scratch (1,734,689 parameters), evaluated on 2,527 unseen test images.

---

## 🏗️ Project Structure

```
medical-forgery-detection/
│
├── app/
│   └── app.py                     # Streamlit web application
│
├── src/
│   ├── data/
│   │   ├── convert_dicom.py       # DICOM → PNG conversion
│   │   ├── organize_dataset.py    # Merge datasets into real/forged
│   │   ├── prepare_final_dataset.py # Train/val/test split
│   │   └── dataset_loader.py      # PyTorch Dataset + DataLoader
│   │
│   ├── models/
│   │   ├── cnn_baseline.py        # 5-block CNN architecture
│   │   └── ela_extractor.py       # Error Level Analysis generator
│   │
│   ├── training/
│   │   ├── train.py               # Training loop
│   │   ├── evaluate.py            # Metrics, confusion matrix, ROC
│   │   └── final_evaluation.py    # Full test evaluation report
│   │
│   └── utils/
│       └── gradcam.py             # Grad-CAM heatmap generation
│
├── models_saved/
│   └── cnn_baseline_best.pth      # Trained model weights
│
├── results/
│   ├── plots/                     # Training curves, ROC, confusion matrix
│   ├── metrics/                   # CSV/TXT evaluation reports
│   └── gradcam_outputs/           # Grad-CAM visualizations
│
├── configs/
│   └── config.yaml                # Project configuration
│
├── requirements.txt
└── README.md
```

> **Note:** The `data/` (dataset) and `venv/` (virtual environment) folders are not included in this repository. See setup instructions below to regenerate them.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd medical-forgery-detection
```

### 2. Create a virtual environment (Python 3.11 required)

PyTorch and other ML libraries are not compatible with Python 3.14+. Use **Python 3.11**.

```bash
py -3.11 -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If PyTorch fails to install or load (`DLL load failed` on Windows), install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), then reinstall PyTorch:
> ```bash
> pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 📦 Dataset Setup

This project combines three public datasets. Download them from Kaggle and place them under `data/raw/`:

| Dataset | Source | Used For |
|---|---|---|
| RSNA Pneumonia Detection Challenge | [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) | Real chest X-rays (DICOM) |
| Medical Image Tamper Dataset | [Kaggle](https://www.kaggle.com/datasets/sjagadeeshgiet/image-tamper-dataset) | Forged + original images (4 forgery types) |
| Brain Tumor MRI Dataset | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | Real + forged MRI scans |

```
data/raw/
├── rsna-pneumonia-detection-challenge/
├── Tamper Dataset/
└── Brain MRI/
```

### Run the preprocessing pipeline (in order)

```bash
# 1. Convert RSNA DICOM files to PNG
python src/data/convert_dicom.py

# 2. Merge all 3 datasets into real_all/ and forged_all/
python src/data/organize_dataset.py

# 3. Create train/val/test split (70/15/15)
python src/data/prepare_final_dataset.py

# 4. Generate ELA maps for all images
python src/models/ela_extractor.py
```

This produces the final dataset structure:

```
data/processed/
├── train/{real,forged}/
├── val/{real,forged}/
├── test/{real,forged}/
└── train_ela/, val_ela/, test_ela/   # ELA maps
```

---

## 🚀 Usage

### Train the model

```bash
python src/training/train.py
```

Trains a 5-block CNN for 10 epochs (Adam optimizer, BCE loss, ReduceLROnPlateau scheduler). Best checkpoint saved to `models_saved/cnn_baseline_best.pth`. Training curves saved to `results/plots/`.

### Evaluate the model

```bash
python src/training/final_evaluation.py
```

Runs the model on the test set and generates a confusion matrix, ROC curve, sample predictions grid, and a full text report in `results/`.

### Generate Grad-CAM visualizations

```bash
python src/utils/gradcam.py
```

### Run the web application

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`. Upload a PNG, JPG, or DICOM medical image to get:
- Real / Forged verdict with confidence score
- ELA map (forensic compression analysis)
- Grad-CAM heatmap (highlights tampered region)
- DICOM metadata panel (if applicable)

---

## 🧠 Model Architecture

```
Input (224×224×3 RGB)
  → Conv Block 1 (32 filters)  → MaxPool → 112×112
  → Conv Block 2 (64 filters)  → MaxPool → 56×56
  → Conv Block 3 (128 filters) → MaxPool → 28×28
  → Conv Block 4 (256 filters) → MaxPool → 14×14
  → Conv Block 5 (512 filters) → MaxPool → 7×7
  → Global Average Pooling → 512-d vector
  → FC(512→256) + Dropout(0.5)
  → FC(256→128) + Dropout(0.3)
  → FC(128→1) + Sigmoid
  → Output: 0 = Real, 1 = Forged
```

Each Conv Block = `Conv2D → BatchNorm → ReLU`. Built entirely from scratch — no pretrained weights (no ResNet, no ImageNet transfer learning).

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Deep Learning:** PyTorch 2.1.2, torchvision
- **Image Processing:** OpenCV, Pillow, scikit-image
- **Medical Imaging:** pydicom, SimpleITK
- **Evaluation:** scikit-learn, Matplotlib, Seaborn
- **Web App:** Streamlit
- **Explainability:** Grad-CAM (custom PyTorch hook implementation)

---

## 📄 License

[Information Not Provided — add your preferred license, e.g. MIT]

---

## 🙋 Author

**Jiya Patel** — B.Tech AI/DS, A.D. Patel Institute of Technology