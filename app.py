import streamlit as st
import pandas as pd
import joblib

# Load the trained model using joblib
model = joblib.load("risk_model.pkl")

# App Title
st.title("📊 AI Financial Risk Score Predictor")
st.write("Upload your financial data file (.csv) to get predicted risk scores.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded file shape:", df.shape)
