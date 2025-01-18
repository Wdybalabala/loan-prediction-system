import streamlit as st
import numpy as np
import pickle
import gdown
import os

# Directory where the model parts are saved
model_parts_dir = 'model_parts'  

# Path to save the reassembled model file
reassembled_model_path = 'rf_model_reassembled.pkl'

# Reassemble the parts into the original model file
with open(reassembled_model_path, 'wb') as output_file:
    part_number = 1
    while True:
        part_file = os.path.join(model_parts_dir, f'part_{part_number}.pkl')
        if not os.path.exists(part_file):
            break  
        
        # Read and write the part to the reassembled model file
        with open(part_file, 'rb') as part_f:
            output_file.write(part_f.read())
        
        print(f"Reassembled part: {part_file}")
        part_number += 1

# Load the reassembled model using pickle
try:
    with open(reassembled_model_path, 'rb') as f:
        rf_model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Streamlit UI
st.title("🏦 Loan Prediction System")
st.write("Enter your details to predict your loan status.")

# Input fields for user data
age = st.number_input("Age", min_value=18, max_value=120, value=30)
dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)
annual_income = st.number_input("Annual Income", min_value=0, max_value=1000000, value=70000)
monthly_expenses = st.number_input("Monthly Expenses", min_value=0, max_value=1000000, value=3500)
credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=750)
existing_loans = st.number_input("Existing Loans", min_value=0, max_value=5, value=1)
total_existing_loan_amount = st.number_input("Total Existing Loan Amount", min_value=0, max_value=100000, value=15000)
outstanding_debt = st.number_input("Outstanding Debt", min_value=0, max_value=100000, value=9000)
loan_amount_requested = st.number_input("Loan Amount Requested", min_value=1000, max_value=1000000, value=20000)
loan_term = st.number_input("Loan Term (in months)", min_value=12, max_value=240, value=12)
interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=15.0, value=10.0)

# Categorical input fields 
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
city_town = st.selectbox("City/Town", ["Urban", "Suburban"])

# Encode categorical variables
marital_status_married = 1 if marital_status == "Married" else 0
marital_status_single = 1 if marital_status == "Single" else 0
employment_status_self_employed = 1 if employment_status == "Self-Employed" else 0
employment_status_unemployed = 1 if employment_status == "Unemployed" else 0
city_town_suburban = 1 if city_town == "Suburban" else 0
city_town_urban = 1 if city_town == "Urban" else 0

# When the user clicks "Predict" button
if st.button("Predict Loan Status"):
    # Prepare the data for prediction
    application_data = np.array([age, dependents, annual_income, monthly_expenses, credit_score,
                                 existing_loans, total_existing_loan_amount, outstanding_debt,
                                 loan_amount_requested, loan_term, interest_rate, 
                                 marital_status_married, marital_status_single, employment_status_self_employed,
                                 employment_status_unemployed, city_town_suburban, city_town_urban]).reshape(1, -1)

    # Make prediction using trained Random Forest
    prediction = rf_model.predict(application_data)

    # Display result
    if prediction == 1:
        st.markdown(
            """
            <div style="border: 2px solid green; background-color: #e6ffe6; padding: 10px; border-radius: 5px;">
                <h4 style="color: green; text-align: center;">Loan Status: Approved</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="border: 2px solid red; background-color: #ffe6e6; padding: 10px; border-radius: 5px;">
                <h4 style="color: red; text-align: center;">Loan Status: Rejected</h4>
            </div>
            """,
            unsafe_allow_html=True
        )


