import streamlit as st
import numpy as np
import joblib
import xgboost

# Load model
model = joblib.load("diabetes_model.pkl")

st.title("Prediksi Diabetes")

# Input 8 fitur utama dari user
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)

# ============================
# Hitung fitur tambahan
# ============================

# Weight Category (One-hot encoding)
weight_categories = {
    "Normal": [0, 0, 0, 0],
    "Obese": [0, 0, 0, 0],
    "Overweight": [0, 0, 0, 0],
    "Underweight": [0, 0, 0, 0]
}

if bmi < 18.5:
    weight_categories["Underweight"] = [0, 0, 0, 1]
elif bmi < 25:
    weight_categories["Normal"] = [1, 0, 0, 0]
elif bmi < 30:
    weight_categories["Overweight"] = [0, 0, 1, 0]
else:
    weight_categories["Obese"] = [0, 1, 0, 0]

# Insulin Level Category (One-hot encoding)
insulin_level_normal = 0
insulin_level_prediabet = 0

if 16 <= insulin <= 166:
    insulin_level_normal = 1
elif insulin > 166:
    insulin_level_prediabet = 1

# ============================
# Gabung semua fitur
# ============================
input_features = [
    pregnancies, glucose, blood_pressure, skin_thickness, insulin,
    bmi, dpf, age,
    *weight_categories["Normal"],
    *weight_categories["Obese"],
    *weight_categories["Overweight"],
    *weight_categories["Underweight"],
    insulin_level_normal,
    insulin_level_prediabet
]

# Pastikan jumlah fiturnya 14
if len(input_features) != 14:
    st.error(f"Jumlah fitur tidak sesuai: {len(input_features)}")
else:
    input_array = np.array([input_features])  # Model butuh shape (1, 14)

    if st.button("Prediksi"):
        prediction = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0][1] * 100

        if prediction == 1:
            st.error(f"Hasil Prediksi: Positif Diabetes (Probabilitas: {prob:.2f}%)")
        else:
            st.success(f"Hasil Prediksi: Negatif Diabetes (Probabilitas: {prob:.2f}%)")
