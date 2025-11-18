from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model_pipeline import train_all_models
from src.preprocess import SCALE_COLUMNS

app = FastAPI(title="Alzheimer's Disease Classifier API", version="1.0.0")

MODEL_PATH = "models/Random_Forest.joblib"
model = None


def load_best_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return True
    return False


@app.on_event("startup")
async def startup_event():
    if not load_best_model():
        print(f"Warning: Model not found at {MODEL_PATH}")


class PredictionInput(BaseModel):
    Age: float
    Gender: float
    Ethnicity: float
    EducationLevel: float
    BMI: float
    Smoking: float
    AlcoholConsumption: float
    PhysicalActivity: float
    DietQuality: float
    SleepQuality: float
    FamilyHistoryAlzheimers: float
    CardiovascularDisease: float
    Diabetes: float
    Depression: float
    HeadInjury: float
    Hypertension: float
    SystolicBP: float
    DiastolicBP: float
    CholesterolTotal: float
    CholesterolLDL: float
    CholesterolHDL: float
    CholesterolTriglycerides: float
    MMSE: float
    FunctionalAssessment: float
    MemoryComplaints: float
    BehavioralProblems: float
    ADL: float
    Confusion: float
    Disorientation: float
    PersonalityChanges: float
    DifficultyCompletingTasks: float
    Forgetfulness: float


class PredictionOutput(BaseModel):
    prediction: int
    diagnosis: str


class RetrainOutput(BaseModel):
    status: str
    message: str


@app.get("/")
async def root():
    return {
        "title": "Alzheimer's Disease Classifier API",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/retrain": "POST - Retrain all models",
            "/health": "GET - Health check",
            "/docs": "Swagger UI documentation"
        }
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    """
    Make a prediction using the best trained model.
    Input: 32 features from the dataset.
    Output: Prediction (0=Healthy, 1=Alzheimer's).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([data.dict()])
        
        cols_to_scale = [c for c in SCALE_COLUMNS if c in input_df.columns]
        if cols_to_scale:
            scaler = StandardScaler()
            input_df[cols_to_scale] = scaler.fit_transform(input_df[cols_to_scale])
        
        prediction = model.predict(input_df)[0]
        diagnosis = "Alzheimer's Disease" if prediction == 1 else "Healthy"

        return PredictionOutput(prediction=int(prediction), diagnosis=diagnosis)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/retrain", response_model=RetrainOutput)
async def retrain():
    """
    Retrain all models using the default dataset.
    This will update all saved models in the models/ directory.
    """
    try:
        print("Starting model retraining...")
        results_df = train_all_models(
            csv_path='alzheimers_disease_data.csv',
            results_dir='results',
            models_dir='models'
        )
        
        if load_best_model():
            return RetrainOutput(
                status="success",
                message="All models retrained and best model reloaded successfully"
            )
        else:
            return RetrainOutput(
                status="error",
                message="Models trained but failed to reload best model"
            )
    except Exception as e:
        print(f"Retrain error: {str(e)}")
        return RetrainOutput(
            status="error",
            message=f"Retraining failed: {str(e)}"
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }
