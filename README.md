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

| Model | MAE | MSE | R² |
|---|---|---|---|
| CatBoost | 5.95 | 123.08 | 0.988 |
| LightGBM | 5.76 | 127.28 | 0.988 |
| XGBoost | 5.91 | 128.24 | 0.988 |
| Decision Tree | 7.18 | 166.91 | 0.984 |
| KNN | 8.40 | 217.99 | 0.980 |
| Linear Regression | 9.49 | 289.74 | 0.973 |
| Random Forest | 11.15 | 391.81 | 0.964 |

**Best model:** CatBoost / LightGBM (comparable top performance).

## 🗂️ Repository Structure

```
AmanFood/
├── AMAN.ipynb              # Main notebook (or AMAN.py if exported as script)
├── data/
│   └── Food_Prices_in_Egypt.parquet   # (or a link/instructions if too large for GitHub)
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook AMAN.ipynb
```

## 📦 Requirements

See `requirements.txt`.

## 👥 Contributors

- [Your name]
- [Teammates' names]

## 📄 License

Add a license of your choice (e.g., MIT) if you want others to be able to reuse this project.
