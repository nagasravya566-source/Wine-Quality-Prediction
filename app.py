import streamlit as st
import pickle
import numpy as np

st.title("Wine Quality Prediction")

with open("model_rf.pkl", "rb") as f:
    model = pickle.load(f)

fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.1, 2.0, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.9, 15.5, 1.9)
chlorides = st.number_input("Chlorides", 0.01, 0.61, 0.076)
free_sulfur = st.number_input("Free Sulfur Dioxide", 1.0, 72.0, 11.0)
total_sulfur = st.number_input("Total Sulfur Dioxide", 6.0, 289.0, 34.0)
density = st.number_input("Density", 0.99, 1.003, 0.9978)
ph = st.number_input("pH", 2.7, 4.0, 3.51)
sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56)
alcohol = st.number_input("Alcohol", 8.4, 14.9, 9.4)

if st.button("Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Quality: {prediction[0]}")
