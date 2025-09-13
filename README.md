# Career Switch Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project designed to predict the likelihood of an employee switching their career. This repository contains the full workflow for data loading, exploratory analysis, feature engineering, model training, and evaluation.

---

## Project Overview

The goal of this project is to classify individuals based on their professional and demographic data into one of two categories:

- **0:** Unlikely to change career
- **1:** Likely to change career

This is a classic **binary classification** problem with a significant **class imbalance**, handled using the **SMOTE oversampling technique**.

---

## Dataset

The dataset (`Career_Switch_Prediction_Dataset.csv`) contains features related to an individual's background:

- **Demographics:** `city`, `gender`
- **Education:** `relevent_experience`, `education_level`, `major_discipline`
- **Professional Experience:** `experience`, `company_size`, `company_type`
- **Job History:** `last_new_job`, `training_hours`
- **Target Variable:** `will_change_career` (0 or 1)

---

## Installation & Setup

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/career-switch-prediction.git
cd career-switch-prediction
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

_Recommended Python version: 3.8+_

---

## Usage

Run the main Jupyter Notebook or Python script:

If using the notebook

```bash
jupyter notebook 04_CSE422_Lab_20_M25.ipynb
```

If using the Python script

```bash
python 04_cse422_lab_20_m25.py
```

The script will:

- Load and preprocess the data
- Perform exploratory data analysis (EDA)
- Train multiple classification models:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Neural Network
- Apply SMOTE to handle class imbalance
- Evaluate all models on a held-out test set
- Generate performance metrics, confusion matrices, and ROC curves
- Perform K-Means clustering on numeric features
- Save the model comparison results to `model_metrics_comparison.csv`
- Save a comprehensive project summary to `project_info.json`

---

## Project Outputs

- `model_metrics_comparison.csv` → Evaluation metrics for each model
- `project_info.json` → Dataset statistics, best model info, and overall results

Visualizations:

- Class distribution chart
- Correlation heatmap
- Confusion matrices
- Performance comparison bar charts
- ROC curves
- K-Means clustering visualization

---

## Customization

- **Preprocessing:** Edit `ordinal_mapper()` and `size_to_num()` for different encodings
- **Models:** Add more models inside the `models` dictionary
- **Hyperparameters:** Tune `n_neighbors`, Neural Network layers, etc., for improved results

---

## Author

Ahnaf Rahman Brinto

Course: CSE422 — Artificial Intelligence Lab

BRAC University

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Acknowledgments

- Dataset provided for academic purposes
- Built with **pandas**, **scikit-learn**, **matplotlib**, **seaborn**, and **tensorflow**
