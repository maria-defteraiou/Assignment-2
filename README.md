# A Repeated Nested Cross-Validation Framework for Breast Cancer Classification  
**MLCB25 – Assignment 2**
---

## Overview  
This project implements a repeated nested cross-validation (rnCV) pipeline for the classification of breast cancer tumors using clinical image-derived features.

## Objectives  
- Perform exploratory data analysis (EDA) on 512 tumor samples 
- Implement an object-oriented rnCV framework  
- Tune, evaluate, and compare multiple classification models  
- Select the best model and train a final deployment-ready instance  

## Dataset  
The dataset `breast_cancer.csv` includes 30 numeric features extracted from fine needle aspirate (FNA) images of breast masses, annotated as benign (B) or malignant (M) (`diagnosis` column).

## Models Evaluated  
- Logistic Regression (Elastic Net)  
- Gaussian Naive Bayes  
- Linear Discriminant Analysis  
- Support Vector Machines  
- Random Forest  
- LightGBM  

## Metrics Used  
- AUC, MCC, F1/F2 Score  
- Precision, Recall, Specificity  
- Balanced Accuracy, PRAUC  
- Confidence Intervals via Bootstrap resampling technique

## Repository Structure  
```
📁 notebooks/       → Jupyter notebooks for EDA and experimentation  
📁 src/             → Source code: rnCV pipeline, utils, model selection  
📁 models/          → Final saved model (.pkl)  
📄 requirements.txt → Python package dependencies  
📄 README.md        → Project description and usage  
📄 report.pdf       → Technical report and results  
```

## Setup Instructions

### 1. Clone the Repository  
```bash
git clone https://github.com/YourUsername/Assignment-2-MLCB25.git
cd Assignment-2-MLCB25
```

### 2. Set Up Environment  
Use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Notebooks  
Launch Jupyter and execute notebooks in order (inside `/notebooks`) to reproduce results.

```bash
jupyter notebook
```

## Dependencies  
See `requirements.txt`. Includes:
- scikit-learn  
- pandas  
- numpy  
- matplotlib, seaborn  
- lightgbm  
- optuna  

## Final Model  
The trained model with optimal hyperparameters is saved in `/models/final_model.pkl`. It can be loaded for predictions on new data.

## Acknowledgments  
This project is part of the **Machine Learning in Computational Biology** course (MLCB25). Dataset and assignment provided by the course organizers.
