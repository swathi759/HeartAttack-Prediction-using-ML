import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("heart.csv")
data.rename(columns={'num': 'target'}, inplace=True)
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Split the data
X = data.drop("target", axis=1)
y = data["target"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# App UI
st.title(" Heart Attack Risk Predictor")
st.write("Enter patient data to check heart disease risk:")

# Input fields
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"])
trestbps = st.number_input("Resting BP (trestbps)", value=120)
chol = st.number_input("Cholesterol (chol)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["No (0)", "Yes (1)"])
restecg = st.selectbox("Resting ECG (restecg)", ["Normal (0)", "ST-T abnormality (1)", "LV hypertrophy (2)"])
thalach = st.number_input("Max Heart Rate (thalach)", value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", ["No (0)", "Yes (1)"])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope of ST segment (slope)", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
ca = st.selectbox("Number of Major Vessels Colored (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", ["Normal (2)", "Fixed defect (1)", "Reversible defect (3)"])

# Convert selections to numbers
sex = 1 if sex == "Male" else 0
cp = int(cp[-2])
fbs = int(fbs[-2])
restecg = int(restecg[-2])
exang = int(exang[-2])
slope = int(slope[-2])
thal = int(thal[-2])

# Predict
if st.button("Predict"):
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error(" Heart Disease Risk Detected!")
    else:
        st.success(" No Heart Disease Risk Detected.")

