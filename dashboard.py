 import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

@st.cache_data
def load_data():
    return pd.read_csv('../data/processed_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('../models/readmission_model.pkl')

def main():
    st.title("Patient Readmission Risk Dashboard")
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    st.header("Data Overview")
    st.write(df.head())
    
    st.header("Feature Distribution")
    feature = st.selectbox("Select a feature", df.columns.drop('readmitted'))
    fig = px.histogram(df, x=feature, nbins=30, title=f"Distribution of {feature}")
    st.plotly_chart(fig)
    
    st.header("Predict Readmission Risk")
    st.write("Enter patient details:")
    
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=20, value=2)
    length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=30, value=5)
    comorbidity_score = st.slider("Comorbidity Score", min_value=0.0, max_value=10.0, value=5.0)
    previous_readmissions = st.number_input("Previous Readmissions", min_value=0, max_value=10, value=1)
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [0 if gender == "Male" else 1],
        'num_procedures': [num_procedures],
        'length_of_stay': [length_of_stay],
        'comorbidity_score': [comorbidity_score],
        'previous_readmissions': [previous_readmissions]
    })
    
    if st.button("Predict"):
        prediction_prob = model.predict_proba(input_data)[0, 1]
        st.write(f"Predicted Readmission Risk Probability: {prediction_prob:.2f}")
    
if __name__ == "__main__":
    main()
