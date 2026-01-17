import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Try importing TensorFlow model loader (may fail on Streamlit Cloud)
try:
    from tensorflow.keras.models import load_model
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

# ---------------- UI ----------------
st.title("💳 Credit Card Fraud Detection (Autoencoder)")
st.write("Upload a trained autoencoder model and a transaction CSV file.")

# Upload files
model_file = st.file_uploader("Upload trained Autoencoder model (.h5)", type=["h5"])
data_file = st.file_uploader("Upload transaction data (.csv)", type=["csv"])

# ---------------- LOGIC ----------------
if data_file is not None:

    # Load CSV
    df = pd.read_csv(data_file)
    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    # Drop unwanted columns if present
    drop_cols = [c for c in ["Amount", "Class"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        st.info(f"Dropped columns: {', '.join(drop_cols)}")

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # ---------------- MODEL HANDLING ----------------
    if model_file is not None and MODEL_AVAILABLE:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(model_file.read())
            model = load_model(tmp.name)

        st.success("✅ Model loaded successfully")

        # Predict
        reconstructed = model.predict(scaled_data)
        reconstruction_error = np.mean(
            np.square(scaled_data - reconstructed), axis=1
        )

        # Threshold
        st.sidebar.title("Threshold Settings")
        threshold = np.percentile(reconstruction_error, 95)

        predictions = (reconstruction_error > threshold).astype(int)

        df["Reconstruction_Error"] = reconstruction_error
        df["Fraud_Prediction"] = predictions

        st.subheader("📊 Prediction Results")
        st.write(df[["Reconstruction_Error", "Fraud_Prediction"]])

        st.success(f"Fraud Transactions: {predictions.sum()}")
        st.info(f"Normal Transactions: {len(predictions) - predictions.sum()}")

        # Plot
        fig, ax = plt.subplots()
        ax.hist(reconstruction_error, bins=50)
        ax.axvline(threshold, linestyle="--")
        ax.set_title("Reconstruction Error Distribution")
        st.pyplot(fig)

    else:
        # ---------------- DEMO MODE ----------------
        st.warning("⚠️ Demo Mode (Model disabled on cloud)")
        st.info("CSV uploaded successfully. App workflow is visible for portfolio.")

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        st.success("✅ App running successfully on Streamlit Cloud")
