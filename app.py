import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

from src.data_pipeline import load_and_preprocess, compute_extreme_thresholds
from src.feature_engineering import engineer_features

# Set page configurations
st.set_page_config(
    page_title="Extreme Weather Prediction",
    page_icon="⛈️",
    layout="wide"
)

# ----------------- App Header -----------------
st.title("⛈️ Extreme Weather Prediction")
st.markdown("""
An end-to-end Machine Learning pipeline predicting extreme rainfall events using 20 years of historical climate data. 
Inspired by the forecasting methodologies of the **Met Office**.
""")

# ----------------- Sidebar Controls -----------------
st.sidebar.header("Location")

# Select location (defaults to Heathrow)
lat = st.sidebar.number_input("Latitude:", min_value = -90.0, max_value = 90.0, value = 51.4833, step=0.0001, format="%.4f")
lon = st.sidebar.number_input("Longitude:", min_value = -180.0, max_value = 180.0, value = -0.45, step=0.0001, format="%.4f")
elv = st.sidebar.number_input("Elevation:", min_value = 0.0, max_value = 10000.0, value = 24.0, step=0.1, format="%.1f")

st.sidebar.header("Model Parameters")

# 1. Interactive Date Range Selector
date_range = st.sidebar.slider(
    "Select Historical Window (Years):",
    min_value=5,
    max_value=20,
    value=20
)

# 2. Interactive Risk Threshold Control
risk_threshold = st.sidebar.slider(
    "Sensitivity / Percentile Threshold (%):",
    min_value=90.0,
    max_value=99.0,
    value=95.0,
    step=0.5,
    help="Higher values define stricter 'extreme' rainfall events"
)

# Automatically calculate dynamic start and end dates relative to today
end = datetime.now()
start = end - timedelta(days=365 * date_range)

# Sidebar information
st.sidebar.markdown("""
---
### Model Information:
*   **Classifier:** Regularised Random Forest
*   **Target Metric:** Recall (TPR)
*   **Extreme Threshold:** Dynamic Monthly Percentile
""")

# ----------------- Main Dashboard -----------------

# Simulate loading and preprocessing data
@st.cache_data # Caches data so changing sidebar options is lightning fast
def get_processed_data(start, end, risk_threshold, lat, lon, elv):
    loc, df_raw = load_and_preprocess(start, end, lat, lon, elv)
    df_thresholded = compute_extreme_thresholds(df_raw, risk_threshold)
    df_final = engineer_features(df_thresholded)
    return loc, df_final

station, df = get_processed_data(start, end, risk_threshold, lat, lon, elv)

st.write(f"### Current Station: **{station}**")

@st.cache_resource
def get_trained_model(df):
    X = df.drop(["extreme_rain"], axis="columns").iloc[:-1]
    y = df["extreme_rain"].iloc[:-1]

    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    rf_model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced_subsample', 
        random_state=42, 
        max_depth=5, 
        min_samples_leaf=5
    )
    rf_model.fit(X_train, y_train)
    return rf_model, X.columns.tolist()

def predict(df, model):
    cd = df.iloc[-1:].drop(["extreme_rain"], axis = "columns")

    prob = model.predict_proba(cd)[0][1]
    pred = model.predict(cd)[0]

    return pred, prob

model, feature_names = get_trained_model(df)

prediction, probability = predict(df, model)

# Extract feature importances and pair with names
importances = model.feature_importances_

# Create a structured DataFrame and sort by importance
df_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)  # Ascending for horizontal bar plot

# Build an interactive Plotly horizontal bar chart
st.subheader("Random Forest Feature Importances")

fig = px.bar(
    df_importance,
    x='Importance',
    y='Feature',
    orientation='h',
    text_auto='.1%',  # Format values as percentages on bars
    title="Predictive Weight by Feature",
    labels={'Importance': 'Relative Importance Weight', 'Feature': 'Predictor Variable'},
    color='Importance',
    color_continuous_scale='Blues'
)

# Clean layout styling
fig.update_layout(
    showlegend=False,
    height=350,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Relative Weight",
    yaxis_title=""
)

# Render in Streamlit
st.plotly_chart(fig, use_container_width=True)



st.subheader("Predict Today's Extreme Rainfall Risk")

st.markdown("---")
res_col1, res_col2 = st.columns(2)

res_col1.metric("Predicted Extreme Risk Probability", f"{probability:.1%}")

if prediction == 1 or probability >= 0.5:
    res_col2.error("**High Risk:** Environmental conditions favor extreme rainfall today.")
else:
    res_col2.success("**Low Risk:** Conditions remain within normal threshold limits.")