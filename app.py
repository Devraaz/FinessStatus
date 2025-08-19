import streamlit as st
import pandas as pd
import numpy as np
import pickle
from statistics import mode

le_g = pickle.load(open('le_g.pkl', 'rb'))
le_s = pickle.load(open('le_s.pkl', 'rb'))
lr = pickle.load(open('lr.pkl', 'rb'))
svc = pickle.load(open('svc.pkl', 'rb'))
rf = pickle.load(open('rf.pkl', 'rb'))
df = pd.read_csv("fitness_dataset.csv")

st.title("🚴 Check your Fitness Status")
st.text("""This is a Streamlit web application that predicts your fitness status based on your daily habits, health indicators, and lifestyle.
It uses multiple machine learning models (Logistic Regression, Support Vector Classifier, and Random Forest) and combines their predictions using a majority voting system.""")

def encode(user_data):

   
    user_data['gender'] = le_g.fit_transform(user_data['gender'].astype(str)) 
    user_data['smokes'] = le_s.fit_transform(user_data['smokes'].astype(str)) 
    
    return user_data

def main():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Enter Age: ", key="age_input", min_value=1, max_value=100, step=1)
        gender = st.selectbox("Enter Gender: ",('M', 'F'))
        height_cm = st.number_input("Enter your Height(in cm): ", key="h_input",  step=1)
        weight_kg = st.number_input("Enter you weight(in kg): ", key="w_input",  step=1)
        heart_rate = st.number_input("Enter Heart Rate: ", key="h_rate",  step=1)

    with col2:
        sleep_hours = st.number_input("Enter Sleep Hour: ", key="s_hr",  step=0.1)
        blood_pressure = st.number_input("Enter Blood Pressure: ", key="bp",  step=0.1)
        nutrition_quality = st.number_input("Enter Nutrition Quality(0-10): ",key="nq", min_value=0.1, max_value=10.0,  step=0.1)
        activity_index = st.number_input("Activity Index(0-5): ",key="act_ind", min_value=0.1, max_value=5.0, step=0.1)
        smokes = st.selectbox("Smoking: ",('yes', 'no'))

    user_data = pd.DataFrame([{
        'age': age,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'heart_rate': heart_rate,
        'blood_pressure': blood_pressure,
        'sleep_hours': sleep_hours,
        'nutrition_quality': nutrition_quality,
        'activity_index': activity_index,
        'smokes': smokes,
        'gender': gender,
    }])

    if st.button('Predict', type='primary'):
        user_data = encode(user_data, )
        st.divider()

        lr_val = lr.predict(user_data)
        svc_val = svc.predict(user_data)
        rf_val = rf.predict(user_data)

        preds = [lr_val[0], svc_val[0], rf_val[0]]
        
        # Show individual predictions
        st.subheader("🔎 Model Predictions")
        st.write(f"Logistic Regression: {'FIT' if lr_val[0] == 1 else 'UNFIT'}")
        st.write(f"SVM: {'FIT' if svc_val[0] == 1 else 'UNFIT'}")
        st.write(f"Random Forest: {'FIT' if rf_val[0] == 1 else 'UNFIT'}")

        # Majority voting
        final_pred = mode(preds)
        message = "FIT" if final_pred == 1 else "UNFIT"

        # Show final decision
        st.subheader("✅ Final Decision (Majority Vote)")
        st.header(f"'{message}'")
main()