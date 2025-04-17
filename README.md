# 🧠 Melanoma Detection with Deep Learning (ISIC Dataset)

A deep learning-based assistant for early **melanoma detection** using dermoscopic images.  
This project uses the **ISIC dataset** (25K+ images) and applies **transfer learning** with `EfficientNet` to classify skin lesions as **benign** or **malignant**.

---

## 📸 Dataset
- **Source:** [ISIC Archive](https://www.isic-archive.com/)
- **Size:** ~25,000 dermoscopic images
- **Labels:** `benign`, `malignant`
- Images are labeled and stored in `data/`, with metadata provided via `metadata.csv`.

---

## 🚀 Model Overview
- 🔍 Uses `EfficientNet-B0` from the `timm` library
- 📦 Fine-tuned on the ISIC dataset (train/val/test split)
- 🎯 Loss: `CrossEntropyLoss`, Optimizer: `Adam`
- 🔥 Implemented in **PyTorch**

---

## 🛠️ Technologies Used
- Python 3.9+
- PyTorch
- torchvision
- timm (for pretrained models)
- pandas, scikit-learn
- ISIC dataset

---

## 🧰 Project Structure


---

## ⚙️ How to Run

1. **Install dependencies:**
```bash
pip install torch torchvision timm pandas scikit-learn


python train_model.py

