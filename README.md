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
├── AMAN.ipynb              # Main notebook
├── data/                   # Place the downloaded dataset here (see below)
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
2. Open and run `AMAN.ipynb`:

```bash
jupyter notebook AMAN.ipynb
```

## 📦 Requirements

See `requirements.txt`.

## 👥 Contributors

- [Your name]
- [Teammates' names]

## 📄 License

Add a license of your choice (e.g., MIT) if you want others to be able to reuse this project.
