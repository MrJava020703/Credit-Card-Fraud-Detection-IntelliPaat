#🛡️ Credit Card Fraud Detection System

Unsupervised Machine Learning using Autoencoders | Streamlit Web App

A production-style machine learning application that detects fraudulent credit card transactions using an Autoencoder neural network.
The system is interactive, scalable, and built for real-world usage with Streamlit UI.

📌 Problem Statement

Credit card fraud is rare but extremely costly.
Traditional rule-based systems fail to detect new and unseen fraud patterns.

👉 This project solves the problem using unsupervised learning, where the model learns normal transaction behavior and flags anomalies automatically.

🧠 Solution Overview

We use an Autoencoder Neural Network trained only on normal transactions.

Normal transactions → Low reconstruction error

Fraudulent transactions → High reconstruction error

A threshold (Auto / Manual) decides whether a transaction is FraUD or NORMAL.

🗂️ Project Directory Structure
fraud-detection-app/
│
├── app.py                  # Streamlit UI & App logic
├── model.py                # Autoencoder architecture & predictions
├── utils.py                # Preprocessing & threshold utilities
├── requirements.txt        # Required Python packages
│
├── data/
│   └── creditcard.csv      # Kaggle dataset
│
├── screenshots/
│   ├── streamlit_ui.png
│   └── error_distribution.png
│
└── README.md

✨ Key Features

✔ Upload your own transaction CSV file
✔ Autoencoder-based anomaly detection
✔ Auto (95th percentile) or Manual threshold selection
✔ Live reconstruction error distribution graph
✔ Simple & clean Streamlit Web UI

⚙️ How the Model Works

Input transactions are scaled & normalized

Autoencoder tries to reconstruct each transaction

Reconstruction error is calculated

Transactions with error above threshold → 🚨 Fraud

Threshold Options

🔹 Auto Mode → 95th Percentile

🔹 Manual Mode → User-defined value

📊 Reconstruction Error Visualization

The app displays a histogram showing:

🟦 Error distribution of transactions

🔴 Threshold line

📌 Fraud cutoff point

This makes the model transparent & explainable.

🖥️ Streamlit Application UI

The interface allows users to:

Upload transaction file

Select threshold type

View total Fraud vs Normal counts

Analyze error distribution in real time

🧪 Dataset Information

📍 Source: Kaggle – Credit Card Fraud Detection

📊 Total Records: 284,807

🚨 Fraud Cases: 492 (0.17%)

🔢 Features: All numeric (PCA transformed)

⚠ Highly imbalanced dataset

🛠️ Technology Stack
Technology	Purpose
Python	Core programming
TensorFlow / Keras	Autoencoder model
Streamlit	Web application
Pandas & NumPy	Data handling
Matplotlib	Visualization
Scikit-learn	Scaling & metrics
🚀 Run the Project Locally
1️⃣ Clone the Repository
git clone https://github.com/your-username/fraud-detection-app.git
cd fraud-detection-app

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Launch the App
streamlit run app.py

📦 Required Libraries (Manual Install)
pip install streamlit pandas numpy matplotlib scikit-learn tensorflow

🎯 Use Cases

Banking fraud monitoring systems

FinTech anomaly detection

Real-time transaction validation

ML portfolio project (LinkedIn / GitHub)

🙌 Acknowledgements

📁 Dataset provided by ULB Machine Learning Group (Kaggle)

💡 Inspired by real-world fraud detection challenges

❤️ Built with passion for Machine Learning & Data Science

⭐ Project Highlights (For LinkedIn)

🚀 Built an end-to-end Credit Card Fraud Detection System using Autoencoders & Streamlit
🧠 Unsupervised ML | Anomaly Detection | Real-time Visualization
