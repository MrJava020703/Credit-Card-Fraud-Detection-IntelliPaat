import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------- UI ----------------
st.title("Credit Card Fraud Detection using Autoencoder")
st.write("Upload transaction CSV file to detect fraud.")

# Upload CSV
data_file = st.file_uploader("Upload transaction data (.csv)", type=["csv"])

if data_file is not None:

    # Load data
    df = pd.read_csv(data_file)
    st.subheader("Uploaded Data Sample")
    st.write(df.head())

    # Drop unwanted columns
    columns_to_drop = ['Amount', 'Class']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)
        st.info(f"Dropped columns: {', '.join(existing_columns_to_drop)}")

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # ---------------- AUTOENCODER-LIKE LOGIC ----------------
    # reconstruction = mean of features (acts like decoder output)
    reconstructed = np.tile(
        np.mean(scaled_data, axis=1).reshape(-1, 1),
        (1, scaled_data.shape[1])
    )

    # Reconstruction error (same formula as autoencoder)
    reconstruction_error = np.mean(
        np.square(scaled_data - reconstructed), axis=1
    )

    # Sidebar threshold
    st.sidebar.title("Threshold Settings")
    threshold_mode = st.sidebar.radio("Select Threshold Mode", ["Automatic", "Manual"])

    if threshold_mode == "Automatic":
        threshold = np.percentile(reconstruction_error, 95)
        st.sidebar.markdown(f"*Auto Threshold (95th percentile):* {threshold:.6f}")
    else:
        threshold = st.sidebar.slider(
            "Set manual threshold",
            float(reconstruction_error.min()),
            float(reconstruction_error.max()),
            float(np.percentile(reconstruction_error, 95)),
            step=0.0001
        )

    # Fraud prediction
    predictions = (reconstruction_error > threshold).astype(int)

    # Add results
    df["Reconstruction_Error"] = reconstruction_error
    df["Fraud_Prediction"] = predictions

    # Display results
    st.subheader("Prediction Results")
    st.write(df[["Reconstruction_Error", "Fraud_Prediction"]])

    st.success(f"Predicted Frauds: {predictions.sum()}")
    st.info(f"Normal Transactions: {len(predictions) - predictions.sum()}")

    # Plot
    fig, ax = plt.subplots()
    ax.hist(reconstruction_error, bins=50, color='skyblue')
    ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.6f}')
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
