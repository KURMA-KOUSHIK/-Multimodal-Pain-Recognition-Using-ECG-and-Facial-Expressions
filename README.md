# ECG-Facial Expression Based Pain Recognition

This project presents a **multi-modal deep learning pipeline** that detects pain levels using facial expressions and ECG signals. By integrating video-based feature extraction and time-series ECG data, this system offers a **robust and objective pain assessment tool** that can be used in clinical and remote health applications.

---

## 📁 Folder Structure

```bash
ecg_facial/
├── facial_model.py       # Facial expression model using ResNet18 + GRU
├── ecg_model.py          # ECG Transformer model for heart rate-based pain classification
├── fusion_model.py       # Fusion logic and evaluation combining facial and ECG models
├── README.md             # Project overview and instructions
```

---

## 🎯 Objective

To develop an automated pain recognition model using synchronized **facial video frames** and **electrocardiogram (ECG)** signals. The system is trained on the **BioVid Heat Pain Database Part A** and classifies pain into four levels:  
- BL1 (no pain)  
- PA1 (mild pain)  
- PA2 (moderate pain)  
- PA3 (severe pain)

---

## ⚙️ Components & Methodology

### ✅ `facial_model.py`
- Custom PyTorch Dataset for facial videos
- Frame-level preprocessing (resize, normalize)
- Uses `ResNet18` (spatial features) + `GRU` (temporal dynamics)
- Trained on video folder hierarchy
- Outputs: pain classification logits (4 classes)

### ✅ `ecg_model.py`
- Preprocesses ECG `.csv` files with filtering and R-peak based heart rate extraction
- Uses a Transformer Encoder to model temporal ECG trends
- Classifies pain based on physiological signal patterns

### ✅ `fusion_model.py`
- Combines predictions from facial and ECG models using weighted logits
- Fusion logic: `80% facial`, `20% ECG`
- Final prediction is made using softmax on the fused logits
- Calculates and prints overall accuracy

---

## 🧪 Dataset: BioVid Heat Pain Database (Part A)
- **87 participants**
- Video frames + synchronized ECG signals
- Pre-annotated with pain levels: BL1, PA1, PA2, PA3
- Used leave-one-subject-out validation protocol for evaluation

---

## 📊 Results
- **Facial model accuracy:** ~70% (approx.)
- **ECG model accuracy:** ~55% (approx.)
- **Fused model accuracy:** **↑ improved with combined context**

> Final accuracy on validation set: **up to 75%** with fusion model

---

## 📚 Technologies Used

- Python, PyTorch
- CNN (ResNet18), GRU, Transformer
- Signal Processing (bandpass filtering, R-R interval)
- Softmax classification, Weighted Fusion

---

## 🏥 Use Case

This system helps in:
- ICU/clinical pain monitoring
- Remote patient care
- Objective pain scoring (non-verbal or unconscious patients)

---

## 📜 License

This code is part of a B.Tech thesis submitted to **Amrita Vishwa Vidyapeetham**.  
Usage for academic or research purposes is permitted. For commercial use, please contact the authors.
