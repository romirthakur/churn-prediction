import streamlit as st

import joblib

import numpy as np

scalervalue = joblib.load("scaler.jpkl")
model = joblib.load("model.jpkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)


gender = st.selectbox("Enter the Gender", ["Male","Female"])



tenure = st.number_input("Enter Tenure", min_value= 0, max_value= 130, value= 10)




monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150, )

gender = st.selectbox("Enter the gender", ["Male", "Female"])

st.divider()

predictbutton = st.button("Predict")

if predictbutton:
    
    gender_selected = 1 if gender == "Female" else 0    
    x = [age, gender_selected, tenure, monthlycharge]
    x_array = np.array(x).reshape(1, -1)

    prediction = model.predict(x_array)[0]
    prediction = "Churn" if prediction == 1 else "No Churn"

    st.write(f"The predicted class is: {prediction}")
else:
    st.write("Please enter the values and hit the predict button for getting a prediction.")