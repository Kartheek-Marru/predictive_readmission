import pandas as pd
import os

def process_data(file_path='../data/synthetic_data.csv'):
    df = pd.read_csv(file_path)
    
    # Convert categorical gender to numeric
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    
    # Additional processing or feature engineering can be added here
    
    return df

def main():
    df = process_data()
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/processed_data.csv', index=False)
    print("Data processed and saved to ../data/processed_data.csv")

if __name__ == "__main__":
    main()
