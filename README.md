import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("risk_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📊 AI Financial Risk Score Predictor")
st.write("Upload your financial data file (.csv) to get predicted risk scores.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded file shape:", df.shape)
        st.write("📄 Columns:", df.columns.tolist())

        # Required input features
        required_features = ["Revenue", "Profit Margin", "Debt Ratio"]
        df_model_input = df[required_features]

        # Predict
        predictions = model.predict(df_model_input)
        df["Risk Score"] = predictions

        # Display results
        st.subheader("📈 Predicted Risk Scores")
        st.write(df)

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv, "risk_scores.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

