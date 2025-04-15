# Optimized Deep Learning Framework for Bone Cancer Detection (ODLF-BCD)

## 📌 Overview

This repository presents the official implementation of the research paper:
**"Enhancing Bone Cancer Detection through Optimized Pre-trained Deep Learning Models and Explainable AI Using the Osteosarcoma-Tumor-Assessment Dataset."**

The proposed system, ODLF-BCD, is a robust, accurate, and interpretable diagnostic framework designed to classify bone cancer in histopathology images. It supports both binary and multi-class classification using pre-trained CNN architectures with advanced optimization and explainability techniques.

## 🚀 Features

- Pre-trained deep learning models: EfficientNet-B4, ResNet50, DenseNet121, InceptionV3, and VGG16.
- Enhanced Bayesian Optimization (EBO) for hyperparameter tuning.
- Grad-CAM and SHAP for Explainable AI integration.
- Transfer learning with custom classification heads.
- Binary and multi-class classification of osteosarcoma.
- Confusion matrix visualizations and training curves (accuracy/loss).
- Ablation study, comparative analysis, and performance benchmarking.

## 🧠 Dataset

- **Name**: Osteosarcoma Tumor Assessment Dataset  
- **Source**: UT Southwestern / UT Dallas  
- **Modality**: Histopathological images annotated for viable and necrotic tumor areas  
- **Preprocessing**: Resize to 224x224, normalization, data augmentation

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn, SHAP

## ⚙️ How to Run

1. Clone the repo:
    ```
    git clone https://github.com/your-username/ODLF-BCD.git
    cd ODLF-BCD
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Download and place the dataset in the `data/` folder.

4. Run training and evaluation:
    ```
    python train.py
    ```

## 📈 Results

- **Binary Classification Accuracy**: 97.9%
- **Multi-Class Classification Accuracy**: 97.3%
- Model interpretability validated with Grad-CAM and SHAP
- Outperforms existing SOTA approaches with improved reliability

## 🧪 Citation

If you use this framework in your research, please cite the paper:

> Bolleddu Devananda Rao, Dr. K. Madhavi, *“Enhancing Bone Cancer Detection through Optimized Pre-trained Deep Learning Models and Explainable AI Using the Osteosarcoma-Tumor-Assessment Dataset,”* [Paper URL will be updated soon].

## 📬 Contact

For queries or collaboration:
- 📧 dev.bolleddu@gmail.com
- 🔗 https://orcid.org/0000-0002-1401-5516
