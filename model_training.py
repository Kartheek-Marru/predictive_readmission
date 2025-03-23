import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os

def load_data(file_path='../data/processed_data.csv'):
    return pd.read_csv(file_path)

def train_model(df, model_type='logistic'):
    # Define features and target
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Choose model type
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type. Choose 'logistic' or 'random_forest'.")
    
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"Model trained. ROC-AUC on test set: {auc:.2f}")
    
    return model

def main():
    df = load_data()
    model = train_model(df, model_type='logistic')
    
    # Save the trained model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/readmission_model.pkl')
    print("Trained model saved to ../models/readmission_model.pkl")

if __name__ == "__main__":
    main()
