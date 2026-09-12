"""
AmanFood Dashboard
------------------
A Streamlit dashboard for exploring Egyptian food price data and getting
price predictions from the trained model produced by AMAN.ipynb.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Expects:
    data/Food_Prices_in_Egypt.parquet   (see README for the download link)
    models/best_model.pkl
    models/scaler.pkl
    models/feature_columns.pkl
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

st.set_page_config(page_title="AmanFood Dashboard", page_icon="🍅", layout="wide")

DATA_PATH = Path("data/Food_Prices_in_Egypt.parquet")
MODEL_PATH = Path("models/best_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
FEATURES_PATH = Path("models/feature_columns.pkl")


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_parquet(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_resource
def load_model_artifacts():
    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and FEATURES_PATH.exists()):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, scaler, feature_columns


df = load_data()
model, scaler, feature_columns = load_model_artifacts()

st.title("🍅 AmanFood – Food Price Dashboard (Egypt)")
st.caption(
    "Explore historical food price patterns across Egypt's governorates, "
    "and get a price estimate from the trained model."
)

if df is None:
    st.error(
        f"Couldn't find the dataset at `{DATA_PATH}`. "
        "Download it from the link in the README and place it in the `data/` folder."
    )
    st.stop()

tab_explore, tab_predict = st.tabs(["📊 Explore the data", "🔮 Predict a price"])

# ---------------------------------------------------------------------------
# Tab 1: Explore
# ---------------------------------------------------------------------------
with tab_explore:
    st.sidebar.header("Filters")

    governorates = sorted(df["Governorate"].dropna().unique().tolist())
    commodities = sorted(df["Commodity"].dropna().unique().tolist())

    selected_governorates = st.sidebar.multiselect(
        "Governorate", governorates, default=governorates
    )
    selected_commodities = st.sidebar.multiselect(
        "Commodity", commodities, default=commodities
    )

    filtered = df[
        df["Governorate"].isin(selected_governorates)
        & df["Commodity"].isin(selected_commodities)
    ]

    if filtered.empty:
        st.warning("No data matches the current filters.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Records", f"{len(filtered):,}")
        col2.metric("Median price (EGP)", f"{filtered['Price_EGP'].median():.1f}")
        col3.metric("Commodities shown", filtered["Commodity"].nunique())

        st.subheader("Price trend over time (median)")
        trend = filtered.groupby("Date", observed=True)["Price_EGP"].median()
        st.line_chart(trend)

        left, right = st.columns(2)

        with left:
            st.subheader("Median price by governorate")
            gov_price = (
                filtered.groupby("Governorate", observed=True)["Price_EGP"]
                .median()
                .sort_values(ascending=False)
            )
            st.bar_chart(gov_price)

        with right:
            st.subheader("Median price by commodity (top 15)")
            commodity_price = (
                filtered.groupby("Commodity", observed=True)["Price_EGP"]
                .median()
                .sort_values(ascending=False)
                .head(15)
            )
            st.bar_chart(commodity_price)

        st.subheader("Price distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(filtered["Price_EGP"].dropna(), bins=50, ax=ax)
        ax.set_xlabel("Price (EGP)")
        st.pyplot(fig)

        with st.expander("Show raw data"):
            st.dataframe(filtered.head(500))

# ---------------------------------------------------------------------------
# Tab 2: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Estimate a food price")

    if model is None:
        st.warning(
            "No trained model found. Run `AMAN.ipynb` end-to-end so it saves "
            "`models/best_model.pkl`, `models/scaler.pkl`, and "
            "`models/feature_columns.pkl`, then reload this dashboard."
        )
    else:
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                governorate = st.selectbox("Governorate", sorted(df["Governorate"].dropna().unique()))
                commodity = st.selectbox("Commodity", sorted(df["Commodity"].dropna().unique()))
                category = st.selectbox("Category", sorted(df["Category"].dropna().unique()))

            with c2:
                region = st.selectbox("Region", sorted(df["Region"].dropna().unique()))
                market_type = st.selectbox("Market type", sorted(df["Market_Type"].dropna().unique()))
                unit = st.selectbox("Unit", sorted(df["Unit"].dropna().unique()))

            with c3:
                season = st.selectbox("Season", sorted(df["Season"].dropna().unique()))
                is_ramadan = st.checkbox("Is Ramadan?")
                month = st.slider("Month", 1, 12, 6)

            c4, c5, c6 = st.columns(3)
            with c4:
                supply_level = st.slider("Supply level", 0.0, 100.0, 70.0)
            with c5:
                demand_level = st.slider("Demand level", 0.0, 100.0, 70.0)
            with c6:
                transport_cost = st.number_input("Transport cost", 0.0, 100.0, 10.0)

            c7, c8, c9 = st.columns(3)
            with c7:
                inflation_index = st.number_input("Inflation index", 0.0, 300.0, 100.0)
            with c8:
                urbanization = st.slider("Urbanization", 0.0, 1.0, 0.7)
            with c9:
                year = st.number_input("Year", 2020, 2035, 2026)

            submitted = st.form_submit_button("Predict price")

        if submitted:
            import numpy as np

            angle = 2 * np.pi * month / 12
            new_row = pd.DataFrame([{
                "Year": year,
                "Month": month,
                "Season": season,
                "Is_Ramadan": int(is_ramadan),
                "Annual_Sin": np.sin(angle),
                "Annual_Cos": np.cos(angle),
                "Governorate": governorate,
                "Region": region,
                "Market_Type": market_type,
                "Unit": unit,
                "Urbanization": urbanization,
                "Commodity": commodity,
                "Category": category,
                "Inflation_Index": inflation_index,
                "Supply_Level": supply_level,
                "Demand_Level": demand_level,
                "Transport_Cost": transport_cost,
            }])

            encoded = pd.get_dummies(
                new_row,
                columns=["Season", "Governorate", "Region", "Commodity", "Market_Type", "Category", "Unit"],
            )
            encoded = encoded.reindex(columns=feature_columns, fill_value=0)
            scaled = scaler.transform(encoded)
            prediction = model.predict(scaled)[0]

            st.success(f"Estimated price: **{prediction:.2f} EGP** per {unit}")
            st.caption(
                "This is a model estimate based on historical patterns, not a live market price."
            )
