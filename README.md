🛡️ Credit Card Fraud Detection 
A smart, interactive system that uses an unsupervised neural network (Autoencoder) to detect fraudulent credit card transactions. Built with ❤️ using Python, TensorFlow, and Streamlit.

📁 Project Structure
fraud-detection-app/
│
├── app.py                   # Main Streamlit app
├── model.py                 # Autoencoder model building & prediction
├── utils.py                 # Data preprocessing, thresholding utils
├── requirements.txt         # Python package requirements
├── data/
│   └── creditcard.csv       # Dataset (Kaggle)
├── screenshots/
│   ├── streamlit_ui.png     # Full UI Screenshot
│   └── error_distribution.png  # Final Graph Output
└── README.md                # Project README
🚀 Features
📤 Upload your own transaction .csv file
🧠 Autoencoder-based fraud prediction
⚙️ Choose Auto or Manual threshold
📊 Live Error Distribution graph output
🖥️ User-friendly Streamlit interface
🧠 How the Model Works
An Autoencoder learns patterns from normal transactions
During prediction, it reconstructs input
If reconstruction error is too high, it's flagged as fraud
Uses 95th percentile or custom threshold to decide cutoff
📈 Reconstruction Error Graph
This graph shows the error distribution of uploaded transactions and highlights the threshold used to identify frauds.

Error Distribution

🔴 Red dashed line = Threshold (Auto: 95th percentile = 0.804494)
🔍 Bars = Frequency of reconstruction errors
🖼️ UI Preview
Here's what the Streamlit interface looks like 👇

Streamlit UI

Shows count of Predicted Frauds and Normal Transactions
Offers threshold options
Displays final visual output in real-time
🧪 Dataset Info
📂 Source: Kaggle – Credit Card Fraud Detection
💡 Records: 284,807
⚠️ Frauds: 492 (~0.17%)
⚙️ Only numeric features; feature scaled for model training
🛠️ Tech Stack
Tool/Library	Role
Python	Core programming
TensorFlow/Keras	Autoencoder implementation
Streamlit	Web interface
Pandas, NumPy	Data manipulation
Matplotlib	Plotting error distribution
⚙️ How to Run Locally
Clone the Repository

git clone https://github.com/your-username/fraud-detection-app.git
cd fraud-detection-app
Install Required Packages

pip install -r requirements.txt
Launch the App

streamlit run app.py
Use the Interface

Upload your CSV ✅
View fraud detection stats 🔍
Analyze reconstruction error plot 📊
📦 Requirements
All dependencies are listed in requirements.txt. To install manually:

pip install streamlit pandas numpy matplotlib scikit-learn tensorflow
🙌 Acknowledgements
💾 Dataset by ULB - Machine Learning Group (Kaggle)
💡 Idea Inspired by real-world fraud challenges
✨ Special thanks to guidance and motivation from peers 💯
About
Detect credit card fraud using ML and autoencoders (with Streamlit UI)

Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Report repository
Releases
No releases published
Packages
No packages published
Footer
