from fastapi import FastAPI
import pandas as pd
from app.infer import infer_model
from pydantic import BaseModel
import logging

#----------------Logging Configuration ------------
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format = "%(asctime)s  - %(levelname)s - %(message)s"
)


app = FastAPI()

# ---------------- Data Models -----------------
class ulogin(BaseModel):
    uname: str
    password: str

class inputs(BaseModel):
    Failed_Transaction_Count_7d     :int
    Risk_Score                      :int
    IP_Address_Flag                 :int
    Transaction_Amount              :int
    Previous_Fraudulent_Activity    :int
    Timestamp                       :str

#------------------ Routes---------------- 
@app.get("/")
def home():
    logging.info("Home endpoint accessed")
    return "Welocme to "

@app.post("/login")
def input(user_login: ulogin ):
    try:
        logging.info(f"login attempt for userid : {user_login.uname}")
        result = "login successfully " if user_login.uname=="1234" and user_login.password =="1234" else "incorrect"
        logging.info(f'Login Result : {result}')
        return result
    except Exception as e:
        logging.info(f"Error during Login: {str(e)}")
        return {"error": "An unexpected error occured during login."}

@app.post("/input")
def user_inp(user_input:inputs):
    try:
        df =   pd.DataFrame([user_input.dict()])
        print(df)
        logging.info(f"Received input values: {df}")
        print(df)
        
        result =infer_model(df)
        print("result :", result[0] )
        logging.info(f"predictiopn result : {result}")
        return {"result": int(result)}   # cast to plain Python int
    except Exception as e:
        logging.error(f"Error during predictio: {str(e)}")
        return {"errro" : "An unexpected error occure during prediction"}

    



