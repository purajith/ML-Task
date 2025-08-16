import pandas as pd
import numpy as np
import joblib
from app.data_preprocessing import  data_processing


def infer_model(df):

    df = data_processing(df)
    print(df)
    selected_features = ['Failed_Transaction_Count_7d',
                        'Risk_Score',
                        'IP_Address_Flag',
                        'Transaction_Amount',
                        'Previous_Fraudulent_Activity',
                        'Hour',
                        'DayOfWeek']
    model =     joblib.load("model/creditcard.pkl")
    inp = df[selected_features]
    pred= model.predict(inp)
    return  pred
