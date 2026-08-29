<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,100:6366f1&height=200&section=header&text=Retail%20Out-of-Stock%20Prediction&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Final%20Capstone%20—%20Code%20Room%20Hub%20AI/ML%20Internship&descAlignY=55&descSize=17" width="100%"/>

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&size=20&pause=1200&color=6366F1&center=true&vCenter=true&width=650&lines=Predicting+tomorrow's+stockouts%2C+today;End-to-end+ML%3A+EDA+%E2%86%92+SHAP+%E2%86%92+API+%E2%86%92+Docker;0.53%25+positive+class+%E2%80%94+handled+the+hard+way" alt="Typing SVG"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-00A98F?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:6366f1,100:0f172a&height=3&width=100%25"/>

## 📖 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Problem Framing](#-problem-framing)
- [Pipeline](#-pipeline)
- [Results](#-results)
- [Explainability (SHAP)](#-explainability-shap)
- [Experiment Tracking](#-experiment-tracking-mlflow)
- [Deployment](#-deployment)
- [Known Limitations](#-known-limitations)
- [Key Learnings](#-key-learnings)
- [Project Structure](#-project-structure)
- [Running It](#-running-it)
- [Tech Stack](#-tech-stack)

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:6366f1,100:0f172a&height=3&width=100%25"/>

## 🧭 Overview

This project predicts whether a store-product combination will run out of stock **the next day**, using historical inventory, sales, and pricing data. It's built as a full end-to-end ML system: data pipeline, model comparison, explainability, experiment tracking, and deployment via a REST API and interactive dashboard.

> **Dataset:** [Retail Store Inventory and Demand Forecasting](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting) (Kaggle) — 76,000 rows covering 5 stores × 20 products over 760 days.

## 🏗️ System Architecture

<p align="center">
  <img src="images/architecture_diagram.png" width="640" alt="System Architecture"/>
</p>

<p align="center"><sub>Raw data → feature engineering → tracked model training → served identically through the API and dashboard → containerized.</sub></p>

## 🎯 Problem Framing

Out-of-stock (OOS) events are rare — only **0.53%** of rows in this dataset have zero inventory.

<p align="center">
  <img src="images/eda_class_imbalance.png" width="520" alt="Class Imbalance"/>
</p>

The project frames this as a **forward-looking binary classification problem**: given today's sales, orders, pricing, and recent trends, predict whether the product will be out of stock **tomorrow**.

<details>
<summary><b>⚠️ A note on data leakage (and why it matters)</b></summary>
<br/>

An early version of this model scored a suspicious ROC-AUC of 1.0. Investigation revealed the target (`Inventory Level == 0`) was mathematically reconstructable from same-day features already in the dataset (`Inventory Level ≈ Previous Inventory + Units Ordered − Units Sold`). This is a common and easy-to-miss trap in inventory/time-series problems: using same-day figures to predict a same-day outcome isn't really *prediction*, it's just re-deriving the answer.

**Fix:** the target was redefined as `Future_OOS` — tomorrow's stock status — using only information that would legitimately be known today. This dropped the ROC-AUC to an honest ~0.94, the real, defensible number reported below.

</details>

## ⚙️ Pipeline

| Step | What happens |
|---|---|
| 1️⃣ EDA | Date range, cardinality checks, class balance, logical consistency checks, category/promotion breakdowns |
| 2️⃣ Feature engineering | Lag features (prev-day inventory/sales), 7-day rolling sales average, day-of-week/month, one-hot categoricals |
| 3️⃣ Preprocessing | Stratified 80/20 split, SMOTE oversampling (train-only, inside CV folds to avoid leakage) |
| 4️⃣ Model comparison | Logistic Regression, Random Forest, LightGBM |
| 5️⃣ Hyperparameter tuning | GridSearchCV, 5-fold CV, via `imblearn` pipeline |
| 6️⃣ Explainability | SHAP TreeExplainer on the final LightGBM model |
| 7️⃣ Experiment tracking | MLflow — all 4 runs logged with params + metrics |
| 8️⃣ Deployment | FastAPI + Streamlit, containerized with Docker |

## 📊 Results

| Model | ROC-AUC | Precision (class 1) | Recall (class 1) | F1 (class 1) |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.846 | 0.14 | 0.10 | 0.12 |
| Random Forest | 0.933 | 0.10 | 0.15 | 0.12 |
| LightGBM (default) | 0.944 | 0.17 | 0.26 | 0.20 |
| **LightGBM (tuned)** ⭐ | **0.938** | 0.16 | 0.15 | 0.15 |

<p align="center">
  <img src="images/model_comparison_chart.png" width="600" alt="Model Comparison"/>
</p>

<sub>Metrics shown at a 0.2 decision threshold (not the default 0.5) — see below.</sub>

**Best hyperparameters** (5-fold CV GridSearchCV): `learning_rate=0.1, max_depth=7, n_estimators=200, num_leaves=31`

<details>
<summary><b>🎚️ Threshold tuning — why 0.2, not 0.5</b></summary>
<br/>

At the default 0.5 probability threshold, models — especially Random Forest — rarely predict the minority class at all, since true stockouts are so rare. Lowering the threshold to 0.2 recovers meaningful recall without collapsing precision. This is a deliberate modeling choice, not a bug: in a real inventory system, missing a stockout (false negative) is usually costlier than a false alarm (false positive), so a lower threshold is the more business-appropriate choice.

</details>

## 🔍 Explainability (SHAP)

<p align="center">
  <img src="images/shap_summary_plot.png" width="640" alt="SHAP Summary Plot"/>
</p>

**Most influential features, in order:**

1. **Previous day's inventory level** — by far the strongest signal; low prior-day stock sharply increases predicted risk
2. **Demand** — higher demand increases risk
3. **Units ordered** — counterintuitively associated with *higher* predicted risk, likely reactive restocking (stores order more when they anticipate running low)
4. **Recent sales trend** (prev-day sales, 7-day rolling avg) — sustained high sales increase risk

Categorical features (region, season, weather, category) have real but comparatively minor influence.

## 🧪 Experiment Tracking (MLflow)

All 4 models were trained and logged as separate MLflow runs, with parameters and metrics tracked for comparison and reproducibility.

<p align="center">
  <img src="images/mlflow_runs_comparison.png" width="640" alt="MLflow Runs"/>
</p>

## 🚀 Deployment

The trained pipeline is served two ways from the same model artifact.

**FastAPI REST API** — a `/predict` endpoint returning an out-of-stock probability + binary prediction.

<p align="center">
  <img src="images/fastapi_swagger_test_request.png" width="440" alt="FastAPI Request"/>
  <img src="images/fastapi_swagger_test_response.png" width="440" alt="FastAPI Response"/>
</p>

**Streamlit Dashboard** — interactive risk gauge, risk band, and plain-language recommendation.

<p align="center">
  <img src="images/streamlit_dashboard.png" width="700" alt="Streamlit Dashboard"/>
</p>

**Docker** — containerized for portable, reproducible deployment.

<p align="center">
  <img src="images/docker_running.png" width="600" alt="Docker Running"/>
</p>

## ⚠️ Known Limitations

Testing the deployed model with extreme low-inventory inputs (e.g. previous inventory of 2 vs. 15) produced nearly identical, very low probabilities. This suggests LightGBM's learned tree splits don't finely distinguish within the very-low-inventory range — likely due to limited training examples in that exact zone. Consistent with the model's modest recall, and an honest limitation rather than a deployment bug.

## 💡 Key Learnings

- **Leakage is easy to miss in inventory/time-series data.** A "perfect" 1.0 ROC-AUC was a red flag, not a win — same-day features algebraically tied to the target will always look too good.
- **Accuracy is the wrong metric for a 0.53%-positive problem.** ROC-AUC and threshold-adjusted precision/recall told a far more honest story than raw accuracy ever could.
- **The right decision threshold depends on business cost, not the default.** Moving from 0.5 → 0.2 was a deliberate trade-off favoring recall.
- **A trained model isn't the finish line.** Getting one pipeline to work consistently across a notebook, REST API, dashboard, and Docker container surfaced real integration issues that never show up during training.

## 📁 Project Structure

```
capstone/
├── images/                     # Diagrams and screenshots used in this README
├── capstone.ipynb              # Full analysis notebook (EDA → modeling → SHAP)
├── app.py                      # FastAPI REST API (/predict endpoint)
├── dashboard.py                # Streamlit interactive dashboard
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
├── oos_prediction_model.pkl    # Trained LightGBM pipeline (SMOTE + model)
├── model_columns.pkl           # Feature column order, for API input alignment
├── sales_data.csv              # Source dataset
└── README.md
```

## ▶️ Running It

**1. API (FastAPI)**
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API docs.

**2. Dashboard (Streamlit)** — with the API running in a separate terminal:
```bash
streamlit run dashboard.py
```

**3. Docker (containerized API)**
```bash
docker build -t oos-prediction-api .
docker run -p 8000:8000 oos-prediction-api
```

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/-pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/-LightGBM-00A98F?style=flat-square)
![SHAP](https://img.shields.io/badge/-SHAP-8A2BE2?style=flat-square)
![MLflow](https://img.shields.io/badge/-MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,100:6366f1&height=100&section=footer" width="100%"/>
