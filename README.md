# Code Repository for: [논문 제목]

This repository contains the code accompanying the article:

**[논문 제목]**  
*Authors: [저자 전체 이름]*  
Published in **PLOS ONE** (2025).  
DOI: [추후 업데이트 예정]

---

## 📖 Overview
This repository provides all source code and scripts used in the study, including:

- **Data preprocessing** (`src/preprocessing.py`)
- **Deep learning models** (`src/model.py`: CNN1D, CNN2D)
- **Training scripts** (`src/train.py`)
- **Evaluation scripts** (`src/evaluation.py`)
- **Notebooks** (`notebooks/analysis.ipynb`: reproduces main figures and results)

---

## ⚙️ Requirements
Experiments were conducted in **Python 3.9** with the following dependencies:

- TensorFlow 2.8.0  
- Keras 2.8.0  
- NumPy 1.23.1  
- Scikit-learn 1.1.0  
- XGBoost 1.7+  

To install all dependencies:
```bash
pip install -r requirements.txt

## 📂 Repository Structure
.
├── data/              # Input data (not included; see below)
├── models/            # Saved model weights (.h5)
├── notebooks/         # Jupyter notebooks for analysis
├── src/               # Source code (preprocessing, models, training, evaluation)
├── requirements.txt   # Python dependencies
└── README.md

## 📊 Data Availability
- The raw data used in this study involve patient-level information and cannot be shared publicly.
- Processed datasets (feature matrices, labels) are available upon reasonable request to the corresponding author, as described in the manuscript’s Data Availability Statement.

## 🚀 How to Run
1. Clone the repository: 
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Create a Python environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Run the main analysis notebook to reproduce results:
```bash
jupyter notebook notebooks/analysis.ipynb

## 📈 Results Reproduction
- Running analysis.ipynb will reproduce the main figures (ROC curves, Grad-CAM visualizations, etc.) reported in the paper.
- Pre-trained models are provided in the models/ directory (e.g., best_model_cnn1d.h5, best_model_cnn2d.h5).

## 📜 License
- This repository is distributed under the MIT License.
- You are free to use, modify, and distribute this code with proper citation.

## ✨ Citation
If you use this code in your research, please cite our paper:
```bash
@article{YourCitationKey,
  title   = {논문 제목},
  author  = {저자1 and 저자2 and ...},
  journal = {PLOS ONE},
  year    = {2025},
  doi     = {https://doi.org/xxxx}
}

## ✉️ Contact
For questions, please contact the corresponding author: 










