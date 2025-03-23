**Predictive Analytics for Patient Readmission Risk**


**Short Description:**  
Machine learning project predicting 30-day patient readmission risk using synthetic healthcare data, complete with data processing, model training, and an interactive Streamlit dashboard.

## Overview

This project demonstrates a predictive analytics approach for identifying patients at high risk of readmission within 30 days. It leverages synthetic healthcare data and applies machine learning techniques to build a predictive model. The repository is structured to showcase a full workflow—from data ingestion and processing, through model training and evaluation, to interactive data visualization via a Streamlit dashboard.

**Key Features:**

- **Data Generation:** Generates synthetic patient data with features such as age, gender, number of procedures, length of stay, comorbidity score, and previous readmissions.
- **Data Processing:** Converts raw data into a format suitable for machine learning, including basic feature engineering.
- **Model Training:** Implements predictive models (e.g., Logistic Regression and Random Forest) using scikit-learn.
- **Evaluation:** Provides detailed performance evaluation using metrics like ROC-AUC and a classification report.
- **Interactive Dashboard:** Uses Streamlit and Plotly to offer an interactive exploration of data and model predictions.
- **Ethical Considerations:** Uses synthetic data to ensure patient privacy. In production, proper HIPAA compliance and data privacy practices must be followed.
