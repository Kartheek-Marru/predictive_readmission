import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def load_data(file_path='../data/processed_data.csv'):
    return pd.read_csv(file_path)

def load_model(model_path='../models/readmission_model.pkl'):
    return joblib.load(model_path)

def evaluate_model(df, model):
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print(f"Overall ROC-AUC: {roc_auc_score(y, y_pred_prob):.2f}")

def main():
    df = load_data()
    model = load_model()
    evaluate_model(df, model)

if __name__ == "__main__":
    main()
