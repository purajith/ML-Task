
from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel
from app.infer import predict_email
app = FastAPI()




class user(BaseModel):
    userid: str
    password : str

class emails(BaseModel):
    email: str

# log in page
@app.get("/")
def home():
    return "Email spam ham Classification"

@app.post("/login")
def login(login: user):
    result = "Is correct" if login.userid == "1234" and login.password == "1234" else "Credential is wrong"
    return {"message": result}

@app.post ("/predict")
def read_item(predict: emails):
    result  = predict_email(predict.email)
    print(result)
    return f"Your mail is: {result}"