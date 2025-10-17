# pylint: disable=E1101
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf 
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


# Load model and preprocessors

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehotencoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit UI

st.set_page_config(page_title="Customer Churn Prediction", page_icon="💼", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #f5f7fa, #c3cfe2);
        font-family: 'Poppins', sans-serif;
    }
    h1 {
        text-align: center;
        color: #2c3e50;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
    }
    .result-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💼 Customer Churn Prediction")


# User input

st.subheader("Enter Customer Details:")

col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=1000, value=600)

with col2:
    balance = st.number_input('🏦 Balance', min_value=0.0, value=1000.0)
    estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, value=50000.0)
    tenure = st.slider('⌛ Tenure', 0, 10, 3)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
    is_active_member = st.selectbox('✅ Is Active Member', [0, 1])


# Prediction Button

if st.button("🔍 Predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine and scale
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Display result
    if prediction_proba > 0.5:
        st.markdown(
            f"<div class='result-card' style='background-color:#ff6b6b;color:white;'>🚨 High Risk: Customer likely to churn!<br>Probability: {prediction_proba:.2f}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card' style='background-color:#2ecc71;color:white;'>✅ Low Risk: Customer likely to stay.<br>Probability: {prediction_proba:.2f}</div>",
            unsafe_allow_html=True
        )
