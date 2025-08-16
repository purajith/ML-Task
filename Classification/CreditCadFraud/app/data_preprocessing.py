import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import FunctionTransformer
import joblib


def data_processing(df):
    df.dropna(inplace=True)

    df.drop_duplicates(inplace=True)

    # Log transforms (must match training)
    df['IP_Address_Flag'] = np.log1p(df['IP_Address_Flag'])
    df['Transaction_Amount'] = np.log1p(df['Transaction_Amount'])
    df['Previous_Fraudulent_Activity'] = np.log1p(df['Previous_Fraudulent_Activity'])

    # Extract datetime features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek


    return df
def train_split(df):
        # Select final feature columns
    selected_features = ['Failed_Transaction_Count_7d',
                    'Risk_Score',
                    'IP_Address_Flag',
                    'Transaction_Amount',
                    'Previous_Fraudulent_Activity',
                    'Hour',
                    'DayOfWeek']
    # Target
    X = df[selected_features]
    y = df['Fraud_Label']

    # Step 4: Train/test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  train_test_split(X, y, test_size=0.2, random_state=42)
def model_building(  X_train, X_test, y_train, y_test):

    # Step 5: Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Step 6: Evaluate
    y_pred = model.predict(X_test)
    
    joblib.dump(model, "model/creditcard.pkl")

    print(" Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return (y_test, y_pred)

if __name__ == "__main__":
    
    df = pd.read_csv("data/synthetic_fraud_dataset.csv")

    data_processing(df)
    X_train, X_test, y_train, y_test =     train_split(df)
    model_building(  X_train, X_test, y_train, y_test)
