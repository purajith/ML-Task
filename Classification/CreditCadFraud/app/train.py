import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import FunctionTransformer
import joblib

from data_preprocessing import data_processing, train_split
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
