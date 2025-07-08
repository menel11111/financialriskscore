import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("risk_model.pkl")

# App Title
st.title("📊 AI Financial Risk Score Predictor")
st.write("Upload your financial data file (.csv) to get predicted risk scores.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        # Read file
        df = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded file shape: {}".format(df.shape))
        st.write("📄 Columns:", df.columns.tolist())

        # Required input features
        required_features = ["Revenue", "Profit Margin", "Debt Ratio"]
        if all(feature in df.columns for feature in required_features):
            df_model_input = df[required_features]

            # Predict
            predictions = model.predict(df_model_input)
            df["Risk Score"] = predictions

            # Display results
            st.subheader("📈 Predicted Risk Scores")
            st.write(df)

            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results as CSV", csv, "risk_scores.csv", "text/csv")
        else:
            st.error(f"Missing required columns: {set(required_features) - set(df.columns)}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
