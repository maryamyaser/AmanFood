# AmanFood – Food Price Prediction in Egypt 🇪🇬

A machine learning project that predicts food commodity prices across Egypt's governorates, using features like location, commodity type, market, supply & demand levels, transport cost, inflation, and seasonality (including Ramadan effects).

## 📊 Project Overview

- **Goal:** Predict `Price_EGP` for food commodities based on economic and geographic factors.
- **Data:** `Food_Prices_in_Egypt.parquet` — includes governorate, commodity, category, market type, supply/demand levels, transport cost, inflation index, and time-based features.
- **Workflow:**
  1. Data loading & cleaning (missing values, duplicates, invalid ranges)
  2. Exploratory Data Analysis (price trends, seasonality, Ramadan effect, governorate/commodity breakdowns)
  3. Preprocessing (dropping leakage-prone columns, one-hot encoding, train/val/test split, feature scaling)
  4. Model training & evaluation
  5. Model comparison
  6. New product price prediction demo

## 🤖 Models Compared

Initial results looked almost too good (R² ≈ 0.97–0.99 across every model) because a leftover feature, `Historical_Median`, correlated 0.98 with the target — every model was mostly just echoing that column. It was removed, and after checking feature importances, most of the remaining predictive power comes from `Commodity` (different foods sit at very different price levels), with supply, demand, transport cost, and inflation refining the estimate within that range.

| Model | MAE | MSE | R² |
|---|---|---|---|
| XGBoost | 6.97 | 155.42 | 0.99 |
| LightGBM | 6.97 | 158.84 | 0.99 |
| Random Forest | 7.43 | 174.55 | 0.98 |
| CatBoost | 7.59 | 180.10 | 0.98 |
| KNN | 8.41 | 218.08 | 0.98 |
| Decision Tree | 11.44 | 317.58 | 0.97 |
| Linear Regression | 10.50 | 436.69 | 0.96 |

**Best model:** XGBoost (tied closely with LightGBM). These numbers will regenerate automatically each time `AMAN.ipynb` is run.

## 🗂️ Repository Structure

```
AmanFood/
├── AMAN.ipynb              # Main notebook
├── app.py                  # Streamlit dashboard
├── data/                   # Place the downloaded dataset here (see below)
├── models/                 # Saved model + scaler + feature columns (created by the notebook)
├── requirements.txt
├── README.md
└── .gitignore
```

## 📥 Dataset

The dataset file is too large for the repository itself, so it's hosted as a GitHub Release asset instead:

**[Download Food_Prices_in_Egypt.parquet](https://github.com/maryamyaser/AmanFood/releases/download/v1.0-data/Food_Prices_in_Egypt.parquet)**

After downloading, place the file inside the `data/` folder so the path matches `data/Food_Prices_in_Egypt.parquet`.

## ⚙️ How to Run

```bash
pip install -r requirements.txt
```

1. Download the dataset from the link above and put it in `data/`.
2. Open and run `AMAN.ipynb` end-to-end — this also saves the trained model into `models/`:

```bash
jupyter notebook AMAN.ipynb
```

## 🖥️ Dashboard

An interactive Streamlit dashboard (`app.py`) lets you explore the data and get price estimates from the trained model.

**Run it locally** (after running the notebook at least once, so `models/` exists):

```bash
streamlit run app.py
```

**Deploy it for free** on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repo to GitHub (already done ✅).
2. Sign in at share.streamlit.io with your GitHub account.
3. Click "New app", pick this repo, and set the main file to `app.py`.
4. Deploy — you'll get a public link to share in your portfolio.



## 📦 Requirements

See `requirements.txt`.

## 👥 Contributors

This project was developed through a specialized division of responsibilities across the machine learning lifecycle:

| Team Member | Project Role | Engineering Scope & Core Responsibilities |
|---|---|---|
| Rowaida Amr Ali | Data Engineering & EDA Lead | Data schema audit, hygiene, and missing-value imputation. Feature engineering (cyclic temporal harmonic signals). Bivariate correlation, distribution skewness, and spatial variance analysis. |
| Basmala El-Husseiny Ismail | Pipeline & Preprocessing Lead | Strict leak-free temporal splitting (Train vs Test). Scaler calibration via Robust & Standard scaling techniques. Categorical encoding and feature matrix alignment. Baseline model initialization and parameter setup. |
| Mariam Yasser Arafat | Advanced ML & Neural Specialist | Training tree ensembles (Random Forest, Decision Trees). Gradient boosting tuning (CatBoost, XGBoost, LightGBM). Neural architecture design (MLPRegressor multi-layer perceptron). Regularization penalty calibration and parameter search. |
| Abdelrahman Emad Ahmed | Evaluation & Benchmark Lead | Multi-model diagnostic evaluation matrix on real currency scales. Generalization gap analysis. Residual distribution analysis, bias-variance trade-offs, and error metrics. Ensemble model construction (VotingRegressor, StackingRegressor). |
| Mohamed Abdullah Sabry | System Architecture & Deployment Lead | Streamlit dashboard development. Real-time model inference pipeline and single-row matrix alignment. Git architecture, model serialization (.pkl), and UI state logic. |


## 📄 License

Add a license of your choice (e.g., MIT) if you want others to be able to reuse this project.
