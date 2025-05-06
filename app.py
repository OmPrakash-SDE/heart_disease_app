import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Heart Disease Predictor", layout="wide", page_icon="🧑‍⚕️")

st.title("Heart Disease Prediction Web App 🧑‍⚕️")

# Load dataset
df = pd.read_csv('heart_disease_data.csv')





# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

accuracy = knn_model.score(X_test, y_test)
st.write(f"Model accuracy on test set: **{accuracy:.2f}**")

# Input fields
st.subheader("Enter Patient Data:")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 0, 120, 50)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    restecg = st.number_input("Resting ECG (0-2)", 0, 2, 0)
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)

with col2:
    sex = st.selectbox("Sex", [0,1])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    slope = st.number_input("Slope (0-2)", 0, 2, 1)
    ca = st.number_input("CA (0-4)", 0, 4, 0)

with col3:
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
    exang = st.selectbox("Exercise Induced Angina", [0,1])
    thal = st.number_input("Thal (1=normal; 2=fixed defect; 3=reversable defect)", 1, 3, 1)

# Prediction
if st.button("Predict Heart Disease"):
    user_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]]
    user_data_scaled = scaler.transform(user_data)
    prediction = knn_model.predict(user_data_scaled)

    if prediction[0] == 1:
        st.error("  Patient has **heart disease**.")
    else:
        st.success("Patient has **no heart disease**.")

