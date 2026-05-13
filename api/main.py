from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib


# =========================
# Load Tuned Model
# =========================
model = joblib.load(
    r"D:\Tự Học\hcm-house-price-prediction\models\best_model.pkl"
)


# =========================
# FastAPI App
# =========================
app = FastAPI(
    title="HCM House Price Prediction API"
)


# =========================
# Input Schema
# =========================
class HouseInput(BaseModel):

    Area: float
    Bedrooms: int
    Bathrooms: int

    District: str

    Latitude: float
    Longitude: float

    Legal: str


# =========================
# Home Route
# =========================
@app.get("/")
def home():

    return {
        "message": "Optuna House Price API Running"
    }


# =========================
# Predict Route
# =========================
@app.post("/predict")
def predict(data: HouseInput):

    input_dict = data.dict()

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]

    return {
        "predicted_price_million_vnd": round(
            float(prediction),
            2
        )
    }