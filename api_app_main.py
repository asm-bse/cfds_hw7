from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
import pandas as pd
from typing import List
from model import train_model

app = FastAPI()

# Define the input data structure
class PredictionInput(BaseModel):
    age: float
    height: float
    weight: float
    aids: int
    cirrhosis: int
    hepatic_failure: int
    immunosuppression: int
    leukemia: int
    lymphoma: int
    solid_tumor_with_metastasis: int

# Endpoint to handle model predictions
@app.post("/predict")
async def predict(input_data: List[PredictionInput]):
    model_filename = "random_forest_model.pkl"
    try:
        # Load the model
        model = joblib.load(model_filename)
    except Exception as e:
        logging.error("Failed to load model", exc_info=True)
        raise HTTPException(status_code=500, detail="Model could not be loaded")

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict() for data in input_data])

    # Make predictions
    try:
        predictions = model.predict(input_df)
    except Exception as e:
        logging.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")

    return {"predictions": predictions.tolist()}

# Endpoint to handle model training
@app.post("/train_model")
async def train_model_endpoint():
    try:
        train_model()
        return {"message": "Model trained successfully"}
    except Exception as e:
        logging.error("Model training failed", exc_info=True)
        return {"error": f"Model training failed: {str(e)}"}
