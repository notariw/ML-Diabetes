import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("diabetes_model.pkl")

st.title("Prediksi Risiko Diabetes - PIMA Dataset")

# Input fitur
pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glukosa", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Tekanan Darah", min_value=0, max_value=140, value=70)
skin_thickness = st.number_input("Tebal Kulit", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Umur", min_value=10, max_value=100, value=30)

if st.button("Prediksi"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Hasil: Berisiko Diabetes")
    else:
        st.success("Hasil: Tidak Berisiko Diabetes")
