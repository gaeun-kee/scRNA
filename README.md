# Code Repository for [Manuscript Under Review at PLOS ONE]

> ⚠️ This repository accompanies a manuscript currently under review at **PLOS ONE**.  
> The code is made publicly available **for peer review and reproducibility purposes**.  
> A final version of this repository will be updated upon acceptance, with DOI and citation details.

---

## 📖 Overview
This repository contains the source code, models, and analysis scripts used in our study:

- **Data preprocessing** (`src/preprocessing.py`)
- **Deep learning models** (`src/model.py` – CNN1D, CNN2D)
- **Training scripts** (`src/train.py`)
- **Evaluation scripts** (`src/evaluate.py`)
- **Notebooks** (`notebooks/analysis.ipynb` – reproduces main results and figures)

---

## ⚙️ Requirements
The experiments were conducted in Python 3.9 with the following major dependencies:

- TensorFlow 2.8.0  
- Keras 2.8.0  
- NumPy 1.23.1  
- Scikit-learn 1.1.0  
- XGBoost (for baseline models)  

To install all dependencies:
```bash
pip install -r requirements.txt
