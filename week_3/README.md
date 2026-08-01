# 🌲 Week 3 – Feature Engineering & Ensemble Learning (Titanic Dataset)

![Status](https://img.shields.io/badge/status-completed-brightgreen)
![Track](https://img.shields.io/badge/track-AI%2FML-blue)
![Internship](https://img.shields.io/badge/Code%20Room%20Hub-Internship-orange)

## 📌 Overview
This task is part of the **AI/ML Track Internship at Code Room Hub**.
The objective was to improve model performance through feature engineering and advanced ensemble algorithms, using the Titanic dataset to predict passenger survival.

## ✅ Objectives
- [x] Clean and prepare the dataset (handle missing values, drop irrelevant columns)
- [x] Encode categorical features and scale numeric features
- [x] Build a Machine Learning pipeline combining preprocessing and modeling
- [x] Train and compare multiple ensemble algorithms
- [x] Perform feature importance analysis on the best-performing model

## 🛠️ Tools & Libraries
`Python` `Scikit-learn` `XGBoost` `LightGBM` `Pandas` `NumPy` `Matplotlib`

<details>
<summary><strong>🤖 Models Trained (click to expand)</strong></summary>

- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

</details>

<details>
<summary><strong>📂 Workflow</strong></summary>

1. Loaded the Titanic dataset and checked for missing values
2. Cleaned the data — filled missing `Age` with median, dropped `Cabin` (too many missing values), dropped `PassengerId`, `Name`, and `Ticket` (no predictive value)
3. One-hot encoded categorical features (`Sex`, `Embarked`)
4. Split data into training (80%) and testing (20%) sets
5. Built an `sklearn` Pipeline combining `StandardScaler` with each model
6. Trained and compared four ensemble models on test accuracy
7. Extracted feature importance from the best model, comparing split-based vs. gain-based importance
8. Visualized feature importance with a bar chart

</details>

## 📊 Feature Importance

![Feature Importance Chart](feature_importance_chart.png)

<details>
<summary><strong>📁 Files in this folder</strong></summary>

| File | Description |
|---|---|
| `Task_Week_3.ipynb` | Notebook with feature engineering, pipeline, and model comparison |
| `titanic.csv` | Dataset used for training and evaluation |
| `feature_importance_chart.png` | Bar chart visualizing feature importance from the best-performing model |

</details>

<details>
<summary><strong>💡 Key Takeaways</strong></summary>

- LightGBM performed best (82.68% accuracy), narrowly ahead of Random Forest (82.12%), with Gradient Boosting and XGBoost close behind (80.45% each)
- `Sex`, `Fare`, and `Age` were the strongest predictors of survival according to gain-based feature importance
- Gain-based importance gave a more meaningful ranking than raw split-count importance — split count favored continuous features like `Age`/`Fare` simply because they had more possible split points, while gain revealed `Sex` as the true top predictor
- Wrapping preprocessing and modeling in a single `Pipeline` made comparing multiple models consistent and leak-safe

</details>

## 🏢 Internship Info
| | |
|---|---|
| **Internship** | AI/ML Track – Code Room Hub |
| **Internship ID** | CRH-2026-ML-014 |
| **Duration** | 1 July 2026 – 30 Aug 2026 |
