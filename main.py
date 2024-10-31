# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# Allow CORS
origins = [
    "http://localhost:5173",  # Update with your React app's URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class InputData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Load your model
model = joblib.load('model.pkl')

@app.get("/")
def hello():
    return "Hello world"

@app.post("/predict")
def predict(input_data: InputData):
    data = [[
        input_data.N,
        input_data.P,
        input_data.K,
        input_data.temperature,
        input_data.humidity,
        input_data.ph,
        input_data.rainfall,
    ]]
    prediction = model.predict(data)
    return {"label": prediction[0]}
