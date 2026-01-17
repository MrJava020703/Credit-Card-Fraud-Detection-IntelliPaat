import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt

# Streamlit UI Title
st.title("Credit Card Fraud Detection using Autoencoder")
st.write("Upload a trained autoencoder model and transaction CSV file to detect fraud.")

# Upload model file
model_file = st.file_uploader("Upload trained Autoencoder model (.h5)", type=["h5"])
# Upload transaction CSV
data_file = st.file_uploader("Upload transaction data (.csv)", type=["csv"])

if model_file and data_file:
    # Load model from temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(model_file.read())
        model = load_model(tmp.name)
        st.success("Model loaded successfully!")

    # Load data
    df = pd.read_csv(data_file)
    st.subheader("Uploaded Data Sample")
    st.write(df.head())

    # Drop unwanted columns if they exist
    columns_to_drop = ['Amount', 'Class']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)
        st.info(f"Dropped columns: {', '.join(existing_columns_to_drop)}")

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Run prediction
    reconstructed = model.predict(scaled_data)
    reconstruction_error = np.mean(np.square(scaled_data - reconstructed), axis=1)

    # Sidebar threshold selection
    st.sidebar.title("Threshold Settings")
    threshold_mode = st.sidebar.radio("Select Threshold Mode", ["Automatic", "Manual"])

    if threshold_mode == "Automatic":
        threshold = np.percentile(reconstruction_error, 95)  # Top 5% flagged as fraud
        st.sidebar.markdown(f"*Auto Threshold (95th percentile):* {threshold:.6f}")
    else:
        threshold = st.sidebar.slider(
            "Set manual threshold", 
            float(reconstruction_error.min()), 
            float(reconstruction_error.max()), 
            float(np.percentile(reconstruction_error, 95)), 
            step=0.0001
        )

    # Generate predictions
    predictions = [1 if e > threshold else 0 for e in reconstruction_error]

    # Add results to DataFrame
    df["Reconstruction_Error"] = reconstruction_error
    df["Fraud_Prediction"] = predictions

    # Display Results
    st.subheader("Prediction Results")
    st.write(df[["Reconstruction_Error", "Fraud_Prediction"]])

    st.success(f"Predicted Frauds: {sum(predictions)}")
    st.info(f"Normal Transactions: {len(predictions) - sum(predictions)}")

    # Plot error distribution
    fig, ax = plt.subplots()
    ax.hist(reconstruction_error, bins=50, color='skyblue')
    ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.6f}')
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
