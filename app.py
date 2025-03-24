import streamlit as st
import numpy as np
import joblib

# Load the trained model and preprocessing objects
model = joblib.load("obesity_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder_gender = joblib.load("label_encoder_gender.pkl")
label_encoder_obesity = joblib.load("label_encoder_obesity.pkl")

# Streamlit UI
st.title("Obesity Prediction Model")

# Input fields
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", label_encoder_gender.classes_)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
physical_activity = st.slider("Physical Activity Level (1-5)", 1, 5, 3)

# Convert categorical input
gender_encoded = label_encoder_gender.transform([gender])[0]

# Prepare input data
input_data = np.array([[age, gender_encoded, height, weight, weight / ((height / 100) ** 2), physical_activity]])
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    predicted_category = label_encoder_obesity.inverse_transform(prediction)[0]
    st.success(f"Predicted Obesity Category: {predicted_category}")
