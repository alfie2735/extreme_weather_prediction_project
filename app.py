import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# Import your custom pipeline modules
from src.data_pipeline import load_and_preprocess, compute_extreme_thresholds
from src.feature_engineering import engineer_features

# Set page configurations
st.set_page_config(
    page_title="UK Extreme Weather Predictor",
    page_icon="⛈️",
    layout="wide"
)

# ----------------- App Header -----------------
st.title("⛈️ Predicting the Unpredictable")
st.markdown("""
An end-to-end Machine Learning pipeline predicting extreme rainfall events using 20 years of historical climate data. 
Inspired by the forecasting methodologies of the **Met Office**.
THIS APP IS STILL UNDER DEVELOPMENT - SOME DATA MAY BE INCORRECTS
""")

# ----------------- Sidebar Controls -----------------
st.sidebar.header("Pipeline Configuration")

# Dropdown for Station Selection
station = st.sidebar.selectbox(
    "Select Weather Station:",
    ["London Heathrow", "Manchester Airport", "Cardiff", "Edinburgh"]
)

st.sidebar.header("Simulation Parameters")

# 1. Interactive Date Range Selector
date_range = st.sidebar.slider(
    "Select Historical Window (Years):",
    min_value=1,
    max_value=20,
    value=10
)

# 2. Interactive Risk Threshold Control
risk_threshold = st.sidebar.slider(
    "Sensitivity / Percentile Threshold (%):",
    min_value=90.0,
    max_value=99.0,
    value=95.0,
    step=0.5,
    help="Higher values define stricter 'extreme' rain events."
)

# Automatically calculate dynamic start and end dates relative to today
end = datetime.now() - timedelta(days=1)  # Yesterday (latest Meteostat Daily update)
start = end - timedelta(days=365 * date_range)

# Sidebar description
st.sidebar.markdown("""
---
### Model Parameters:
*   **Classifier:** Regularised Random Forest
*   **Target Metric:** Recall (TPR)
*   **Extreme Threshold:** Dynamic Monthly Percentile
""")

# ----------------- Main Dashboard -----------------
st.write(f"### Current Station Analysis: **{station}**")

# Simulate loading and preprocessing data
@st.cache_data # Caches data so changing sidebar options is lightning fast
def get_processed_data(start, end, risk_threshold):
    # In practice, point this to your filtered master dataset or individual files
    df_raw = load_and_preprocess(start, end)
    df_thresholded = compute_extreme_thresholds(df_raw, risk_threshold)
    df_final = engineer_features(df_thresholded)
    return df_final

df = get_processed_data(start, end, risk_threshold)

@st.cache_resource
def get_trained_model(df):
    X = df.drop(["extreme_rain"], axis="columns")
    y = df["extreme_rain"]

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

model, feature_names = get_trained_model(df)

# 3. Extract feature importances and pair with names
importances = model.feature_importances_

# Create a structured DataFrame and sort by importance
df_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)  # Ascending for horizontal bar plot

# 4. Build an interactive Plotly horizontal bar chart
st.subheader("Random Forest Feature Importances")
st.caption("Extracted directly from the saved `.joblib` model weights:")

fig = px.bar(
    df_importance,
    x='Importance',
    y='Feature',
    orientation='h',
    text_auto='.1%',  # Format values as percentages on bars
    title="Predictive Weight by Feature",
    labels={'Importance': 'Relative Importance Weight', 'Feature': 'Predictor Variable'},
    color='Importance',
    color_continuous_scale='Viridis'
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

# 2. Interactive Columns for Visualisations
chart_col, data_col = st.columns([2, 1])

with chart_col:
    st.subheader("Feature Importances")
    # Generate Feature Importance Chart dynamically
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Simulated or loaded feature weights
    features = ['Current Pressure', '48h Pressure Lag', 'Relative Humidity', 'Sine Day of Year', 'Cosine Day of Year']
    importances = [0.35, 0.22, 0.18, 0.15, 0.10]
    
    sns.barplot(x=importances, y=features, palette="viridis", ax=ax)
    ax.set_xlabel("Relative Importance Weight")
    plt.tight_layout()
    st.pyplot(fig)


with data_col:
    st.subheader("Engineered Dataset Feature Sample")
    st.write("Inspect the inputs sent to the Random Forest model:")
    cols_to_show = ['pres', 'pres_lag_1', 'pres_lag_2', 'extreme_rain']
    st.dataframe(df[cols_to_show].head(10), use_container_width=True)