# Customer Churn Prediction using ANN 🎯

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://ann-classification-churn-prediction-hv9c7vmh4wrdmuxrebyvcx.streamlit.app/)

A web app that predicts whether a bank’s customer is likely to churn (leave) using an Artificial Neural Network (ANN) built with TensorFlow/Keras, and deployed via Streamlit Cloud.

---

## 🌐 Live Demo

🔗 Access the deployed app here:  
[https://ann-classification-churn-prediction-hv9c7vmh4wrdmuxrebyvcx.streamlit.app/](https://ann-classification-churn-prediction-hv9c7vmh4wrdmuxrebyvcx.streamlit.app/)

---

## 📘 Project Overview

This project implements a predictive model for **customer churn** based on customer demographics, financial behavior, and bank product usage. The model is packaged into a Streamlit web interface that allows users to input custom customer data and receive a real-time churn prediction with probability and a styled result display.

---

## 🛠 Tech Stack

| Component        | Technology / Library             |
|------------------|----------------------------------|
| Web UI / App     | Streamlit                        |
| Backend / ML     | TensorFlow, Keras                |
| Data Preprocessing | scikit-learn (LabelEncoder, OneHotEncoder, StandardScaler) |
| Serialization    | Pickle                           |
| Deployment        | Streamlit Cloud                  |
| Language          | Python 3.10+                     |

---

## 🗂️ Project Structure
│
├── app.py # Streamlit application code
├── model.h5 # Trained ANN model
├── scaler.pkl # Pickled StandardScaler
├── label_encoder_gender.pkl # Pickled LabelEncoder for gender
├── onehotencoder.pkl # Pickled OneHotEncoder for geography
├── experiments.ipynb # Notebook where model is trained / tuned
├── prediction.ipynb # Notebook to test model predictions
├── requirements.txt # Required Python packages
└── README.md # This documentation file


---

## 📊 Data & Model Details

**Dataset Features (inputs):**

- `CreditScore`  
- `Geography` (one-hot encoded)  
- `Gender` (label encoded)  
- `Age`  
- `Tenure`  
- `Balance`  
- `NumOfProducts`  
- `HasCrCard`  
- `IsActiveMember`  
- `EstimatedSalary`

**Target Label:**

- `Exited` (1 = churned, 0 = stayed)

**Model Architecture:**

- Input layer with all features  
- 2 hidden dense layers (e.g. 6 neurons each, ReLU)  
- Output layer with 1 neuron and sigmoid activation  
- Trained using binary crossentropy loss and an optimizer like Adam  

---

## 🧩 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Customer-Churn-Prediction-ANN.git
   cd Customer-Churn-Prediction-ANN
2. **Create & activate a virtual environment**
   python -m venv venv
  # On Linux / macOS
  source venv/bin/activate
  # On Windows
  venv\Scripts\activate
3. **Install dependencies**
  pip install -r requirements.txt
4. **Launch the Streamlit app**
  streamlit run app.py
5. **Open browser and interact**
  Visit http://localhost:8501 to view and use the app locally.



