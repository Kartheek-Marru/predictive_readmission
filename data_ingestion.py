import numpy as np
import pandas as pd
import os

def generate_synthetic_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    # Simulate basic patient features
    age = np.random.randint(18, 90, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    num_procedures = np.random.poisson(2, n_samples)
    length_of_stay = np.random.randint(1, 15, n_samples)
    comorbidity_score = np.random.uniform(0, 10, n_samples)
    previous_readmissions = np.random.poisson(1, n_samples)
    
    # Create a synthetic target variable for 30-day readmission risk
    risk_score = (age/90) + (num_procedures/10) + (length_of_stay/15) + (comorbidity_score/10) + (previous_readmissions/5)
    probability = 1 / (1 + np.exp(-risk_score))
    readmitted = np.random.binomial(1, probability)
    
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'num_procedures': num_procedures,
        'length_of_stay': length_of_stay,
        'comorbidity_score': comorbidity_score,
        'previous_readmissions': previous_readmissions,
        'readmitted': readmitted
    })
    
    return data

def main():
    data = generate_synthetic_data(n_samples=1000)
    
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    data.to_csv('../data/synthetic_data.csv', index=False)
    print("Synthetic data generated and saved to ../data/synthetic_data.csv")

if __name__ == "__main__":
    main()
