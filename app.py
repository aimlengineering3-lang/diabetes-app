import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =======================
# Title
# =======================
st.title("🩺 Diabetes Prediction App")

# =======================
# Load Dataset
# =======================
df = pd.read_csv("diabetes.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# =======================
# Handle Missing Values
# =======================
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols:
    df[col] = df[col].replace(0, df[col].mean())

# =======================
# EDA Section
# =======================
st.subheader("📊 Data Visualization")

if st.checkbox("Show Histogram"):
    fig, ax = plt.subplots()
    df.hist(ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Correlation Matrix"):
    corr = df.corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    st.pyplot(fig)

# =======================
# Features & Target
# =======================
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# =======================
# Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# Scaling
# =======================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =======================
# Train Model
# =======================
model = LogisticRegression()
model.fit(X_train, y_train)

# =======================
# Model Accuracy
# =======================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"Accuracy: {acc:.2f}")

# =======================
# User Input Section
# =======================
st.subheader("🧾 Enter Patient Details")

preg = st.number_input("Pregnancies", 0, 20)
glu = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
ins = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# =======================
# Prediction Button
# =======================
if st.button("Predict"):
    new_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    new_data = scaler.transform(new_data)

    prediction = model.predict(new_data)

    if prediction[0] == 1:
        st.error("❌ Person is Diabetic")
    else:
        st.success("✅ Person is Not Diabetic")