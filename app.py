from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model_pipeline import train_all_models
from src.preprocess import SCALE_COLUMNS

app = FastAPI(
    title="Alzheimer's Disease Classifier API",
    version="1.0.0",
    description="""
    ## Alzheimer's Disease Classification API
    
    This API provides machine learning-based prediction for Alzheimer's disease diagnosis.
    
    ### Features:
    * **Prediction**: Classify patients as Healthy or Alzheimer's Disease using K-Nearest Neighbors model
    * **Model Retraining**: Retrain all models with updated data
    * **Health Check**: Monitor API and model status
    
    ### Model Information:
    * **Algorithm**: Best performing model (automatically selected based on F1-score)
    * **Available Models**: K-Nearest Neighbors, Random Forest, Decision Tree, Logistic Regression, SVM, AdaBoost
    * **Input Features**: 32 clinical and demographic features
    * **Output**: Binary classification (0=Healthy, 1=Alzheimer's Disease)
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/best_model.joblib"  # Use best model instead of hardcoded KNN
SCALER_PATH = "models/scaler.joblib"
model = None
scaler = None
expected_columns = None  # Will store the expected column order from training data
model_name = None  # Will store the name of the best model


def load_model():
    """Charge le meilleur modèle et le scaler."""
    global model, scaler, expected_columns, model_name
    
    # Charger le meilleur modèle
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            # Try to get model name from results
            try:
                results_df = pd.read_csv('results/results.csv')
                if not results_df.empty:
                    model_name = results_df.iloc[0]['Model']  # Best model is first (sorted by F1-score)
                    print(f"Best model loaded: {model_name} from {MODEL_PATH}")
                else:
                    model_name = "Best Model"
                    print(f"Model loaded from {MODEL_PATH}")
            except:
                model_name = "Best Model"
                print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
        # Fallback to KNN if best_model doesn't exist
        fallback_path = "models/K-Nearest_Neighbors.joblib"
        if os.path.exists(fallback_path):
            try:
                model = joblib.load(fallback_path)
                model_name = "K-Nearest Neighbors"
                print(f"Fallback: Loaded KNN model from {fallback_path}")
            except Exception as e:
                print(f"Failed to load fallback model: {e}")
                return False
        else:
            return False
    
    # Charger le scaler s'il existe, sinon le recréer
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded from {SCALER_PATH}")
        except Exception as e:
            print(f"Failed to load scaler: {e}")
            print("Will recreate scaler from training data...")
            scaler = _create_scaler()
    else:
        print(f"Scaler not found at {SCALER_PATH}, creating from training data...")
        scaler = _create_scaler()
    
    # Get expected column order from training data
    if expected_columns is None:
        try:
            from model_pipeline import prepare_data
            X_train, _, _, _, _ = prepare_data(
                csv_path='alzheimers_disease_data.csv',
                test_size=0.2,
                random_state=40
            )
            expected_columns = X_train.columns.tolist()
            print(f"Expected columns order set: {len(expected_columns)} columns")
        except Exception as e:
            print(f"Warning: Could not set expected columns: {e}")
    
    return True


def _create_scaler():
    """Crée le scaler à partir des données d'entraînement."""
    global expected_columns
    try:
        from model_pipeline import prepare_data
        X_train, _, _, _, new_scaler = prepare_data(
            csv_path='alzheimers_disease_data.csv',
            test_size=0.2,
            random_state=40
        )
        # Store the expected column order
        expected_columns = X_train.columns.tolist()
        print("Scaler created successfully from training data")
        print(f"Expected columns order: {expected_columns}")
        return new_scaler
    except Exception as e:
        print(f"Failed to create scaler: {e}")
        return None


@app.on_event("startup")
async def startup_event():
    if not load_model():
        print(f"Warning: Failed to load model from {MODEL_PATH}")


class PredictionInput(BaseModel):
    """Input model for prediction request with all required features."""
    
    Age: float = Field(..., description="Patient age in years", example=65.0)
    Gender: float = Field(..., description="Gender (0=Female, 1=Male)", example=1.0)
    Ethnicity: float = Field(..., description="Ethnicity code", example=2.0)
    EducationLevel: float = Field(..., description="Education level (0-10 scale)", example=4.0)
    BMI: float = Field(..., description="Body Mass Index", example=25.5)
    Smoking: float = Field(..., description="Smoking status (0=No, 1=Yes)", example=0.0)
    AlcoholConsumption: float = Field(..., description="Alcohol consumption level (0-5 scale)", example=2.0)
    PhysicalActivity: float = Field(..., description="Physical activity level (0-5 scale)", example=3.0)
    DietQuality: float = Field(..., description="Diet quality score (0-10 scale)", example=7.0)
    SleepQuality: float = Field(..., description="Sleep quality score (0-10 scale)", example=6.0)
    FamilyHistoryAlzheimers: float = Field(..., description="Family history of Alzheimer's (0=No, 1=Yes)", example=0.0)
    CardiovascularDisease: float = Field(..., description="Cardiovascular disease (0=No, 1=Yes)", example=0.0)
    Diabetes: float = Field(..., description="Diabetes status (0=No, 1=Yes)", example=0.0)
    Depression: float = Field(..., description="Depression status (0=No, 1=Yes)", example=0.0)
    HeadInjury: float = Field(..., description="History of head injury (0=No, 1=Yes)", example=0.0)
    Hypertension: float = Field(..., description="Hypertension status (0=No, 1=Yes)", example=0.0)
    SystolicBP: float = Field(..., description="Systolic blood pressure (mmHg)", example=130.0)
    DiastolicBP: float = Field(..., description="Diastolic blood pressure (mmHg)", example=85.0)
    CholesterolTotal: float = Field(..., description="Total cholesterol (mg/dL)", example=200.0)
    CholesterolLDL: float = Field(..., description="LDL cholesterol (mg/dL)", example=120.0)
    CholesterolHDL: float = Field(..., description="HDL cholesterol (mg/dL)", example=50.0)
    CholesterolTriglycerides: float = Field(..., description="Triglycerides (mg/dL)", example=150.0)
    MMSE: float = Field(..., description="Mini-Mental State Examination score (0-30)", example=25.0)
    FunctionalAssessment: float = Field(..., description="Functional assessment score (0-10 scale)", example=7.0)
    MemoryComplaints: float = Field(..., description="Memory complaints (0=No, 1=Yes)", example=0.0)
    BehavioralProblems: float = Field(..., description="Behavioral problems (0=No, 1=Yes)", example=0.0)
    ADL: float = Field(..., description="Activities of Daily Living score (0-10 scale)", example=5.0)
    Confusion: float = Field(..., description="Confusion symptoms (0=No, 1=Yes)", example=0.0)
    Disorientation: float = Field(..., description="Disorientation symptoms (0=No, 1=Yes)", example=0.0)
    PersonalityChanges: float = Field(..., description="Personality changes (0=No, 1=Yes)", example=0.0)
    DifficultyCompletingTasks: float = Field(..., description="Difficulty completing tasks (0=No, 1=Yes)", example=0.0)
    Forgetfulness: float = Field(..., description="Forgetfulness symptoms (0=No, 1=Yes)", example=0.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "Age": 65,
                "Gender": 1,
                "Ethnicity": 2,
                "EducationLevel": 4,
                "BMI": 25.5,
                "Smoking": 0,
                "AlcoholConsumption": 2,
                "PhysicalActivity": 3,
                "DietQuality": 7,
                "SleepQuality": 6,
                "FamilyHistoryAlzheimers": 0,
                "CardiovascularDisease": 0,
                "Diabetes": 0,
                "Depression": 0,
                "HeadInjury": 0,
                "Hypertension": 0,
                "SystolicBP": 130,
                "DiastolicBP": 85,
                "CholesterolTotal": 200,
                "CholesterolLDL": 120,
                "CholesterolHDL": 50,
                "CholesterolTriglycerides": 150,
                "MMSE": 25,
                "FunctionalAssessment": 7,
                "MemoryComplaints": 0,
                "BehavioralProblems": 0,
                "ADL": 5,
                "Confusion": 0,
                "Disorientation": 0,
                "PersonalityChanges": 0,
                "DifficultyCompletingTasks": 0,
                "Forgetfulness": 0
            }
        }


class PredictionOutput(BaseModel):
    """Output model for prediction response."""
    prediction: int = Field(..., description="Prediction result (0=Healthy, 1=Alzheimer's Disease)", example=0)
    diagnosis: str = Field(..., description="Human-readable diagnosis", example="Healthy")
    probabilities: dict = Field(None, description="Prediction probabilities for each class", example={"healthy": 0.95, "alzheimer": 0.05})
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "diagnosis": "Healthy"
            }
        }


class RetrainOutput(BaseModel):
    """Output model for retrain response."""
    status: str = Field(..., description="Status of retraining operation", example="success")
    message: str = Field(..., description="Detailed message about the retraining result", 
                        example="All models retrained and KNN model reloaded successfully")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "All models retrained and KNN model reloaded successfully"
            }
        }


@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint providing API information and available endpoints.
    
    Returns basic information about the API and links to all available endpoints.
    """
    return {
        "title": "Alzheimer's Disease Classifier API",
        "version": "1.0.0",
        "description": "Machine learning API for Alzheimer's disease classification",
        "endpoints": {
            "/predict": "POST - Make a prediction using the best performing model",
            "/retrain": "POST - Retrain all models with updated data",
            "/health": "GET - Health check and model status",
            "/docs": "Swagger UI interactive documentation",
            "/redoc": "ReDoc alternative documentation",
            "/openapi.json": "OpenAPI schema in JSON format"
        }
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(data: PredictionInput):
    """
    Make a prediction for Alzheimer's disease classification.
    
    This endpoint uses the best performing model (selected based on F1-score) to classify
    a patient as either Healthy (0) or having Alzheimer's Disease (1) based on
    32 clinical and demographic features.
    
    **Input Requirements:**
    - All 32 features must be provided as float values
    - Features are automatically scaled using the training data scaler
    
    **Output:**
    - `prediction`: Integer (0=Healthy, 1=Alzheimer's Disease)
    - `diagnosis`: Human-readable diagnosis string
    
    **Example Request:**
    See the example in the request body schema below or use the example_request.json file.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        global expected_columns
        
        # Get the data as dict and ensure correct column order
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Ensure columns are in the same order as training data
        if expected_columns is None:
            # Get expected column order from training data
            from model_pipeline import prepare_data
            X_train, _, _, _, _ = prepare_data(
                csv_path='alzheimers_disease_data.csv',
                test_size=0.2,
                random_state=40
            )
            expected_columns = X_train.columns.tolist()
        
        # Reorder input_df to match training data column order
        input_df = input_df[expected_columns]

        # CRITICAL: Scale columns in the EXACT same order as during training
        # The scaler was fitted on columns in SCALE_COLUMNS order, so we must transform in that same order
        # IMPORTANT: Extract columns in SCALE_COLUMNS order, not in input_df column order!
        cols_to_scale = [c for c in SCALE_COLUMNS if c in input_df.columns]
        
        if cols_to_scale:
            if scaler is not None:
                # Extract columns in the EXACT order the scaler expects (SCALE_COLUMNS order)
                # This is critical - if columns are in wrong order, features get scaled incorrectly!
                scaled_data = input_df[cols_to_scale].values  # Get as numpy array in correct order
                scaled_values = scaler.transform(scaled_data)
                # Assign back in the same order
                for i, col in enumerate(cols_to_scale):
                    input_df[col] = scaled_values[:, i]
            else:
                # fallback: scale single row using a fresh scaler
                print("WARNING: Using fallback scaler - this may cause incorrect predictions!")
                tmp_scaler = StandardScaler()
                scaled_data = input_df[cols_to_scale].values
                scaled_values = tmp_scaler.fit_transform(scaled_data)
                for i, col in enumerate(cols_to_scale):
                    input_df[col] = scaled_values[:, i]

        # Debug: Print comprehensive diagnostic information
        print(f"\n{'='*60}")
        print(f"PREDICTION DEBUG INFO")
        print(f"{'='*60}")
        print(f"Model type: {type(model).__name__}")
        print(f"Input shape: {input_df.shape}")
        print(f"Columns ({len(input_df.columns)}): {list(input_df.columns)}")
        print(f"Columns to scale: {cols_to_scale}")
        
        # Print key features before and after scaling
        key_features = ['MMSE', 'MemoryComplaints', 'Age', 'FunctionalAssessment', 'ADL']
        print(f"\nKey features (before scaling):")
        for feat in key_features:
            if feat in input_dict:
                print(f"  {feat}: {input_dict[feat]}")
        
        if cols_to_scale and scaler is not None:
            print(f"\nKey features (after scaling):")
            for feat in key_features:
                if feat in input_df.columns:
                    print(f"  {feat}: {input_df[feat].values[0]:.4f}")
        
        # Check if model is a Pipeline (from sklearn Pipeline wrapper)
        from sklearn.pipeline import Pipeline
        actual_model = model
        if isinstance(model, Pipeline):
            print(f"Model is wrapped in Pipeline with steps: {[s[0] for s in model.steps]}")
            # Get the actual model from the pipeline
            if 'model' in model.named_steps:
                actual_model = model.named_steps['model']
                print(f"Extracted model type: {type(actual_model).__name__}")
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        print(f"\nRaw prediction: {prediction} ({'Alzheimer' if prediction == 1 else 'Healthy'})")
        
        # Get prediction probabilities if available
        try:
            prediction_proba = model.predict_proba(input_df)[0]
            print(f"Prediction probabilities: [Healthy (0): {prediction_proba[0]:.4f}, Alzheimer (1): {prediction_proba[1]:.4f}]")
            
            # Check if probabilities suggest a different prediction
            if prediction_proba[1] > 0.5 and prediction == 0:
                print(f"⚠️  WARNING: Probabilities suggest Alzheimer ({prediction_proba[1]:.4f}) but prediction is Healthy!")
                print(f"   This indicates a potential issue with the model or threshold!")
            elif prediction_proba[0] > 0.5 and prediction == 1:
                print(f"⚠️  WARNING: Probabilities suggest Healthy ({prediction_proba[0]:.4f}) but prediction is Alzheimer!")
        except Exception as e:
            print(f"Could not get probabilities: {e}")
        
        # For some models, check decision function
        try:
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(input_df)[0]
                print(f"Decision function value: {decision:.4f} (positive = Alzheimer, negative = Healthy)")
        except:
            pass
        
        # Check input data statistics
        print(f"\nInput data statistics:")
        print(f"  Min value: {input_df.min().min():.4f}")
        print(f"  Max value: {input_df.max().max():.4f}")
        print(f"  Mean value: {input_df.mean().mean():.4f}")
        print(f"  Contains NaN: {input_df.isna().any().any()}")
        print(f"  Contains Inf: {input_df.isin([float('inf'), float('-inf')]).any().any()}")
        
        print(f"{'='*60}\n")
        
        diagnosis = "Alzheimer's Disease" if prediction == 1 else "Healthy"
        
        # Get prediction probabilities if available
        probabilities = None
        try:
            proba = model.predict_proba(input_df)[0]
            probabilities = {
                'healthy': float(proba[0]),
                'alzheimer': float(proba[1])
            }
        except:
            pass

        return PredictionOutput(
            prediction=int(prediction), 
            diagnosis=diagnosis,
            probabilities=probabilities
        )
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/retrain", response_model=RetrainOutput, tags=["Model Management"])
async def retrain():
    """
    Retrain all machine learning models using the default dataset.
    
    This endpoint retrains all models (KNN, Random Forest, Decision Tree, Logistic Regression, SVM, AdaBoost)
    using the data from `alzheimers_disease_data.csv`. After training, the
    best performing model (by F1-score) is automatically reloaded for use in predictions.
    
    **Process:**
    1. Loads training data from the CSV file
    2. Trains all models (KNN, Random Forest, Decision Tree, Logistic Regression, SVM, AdaBoost)
    3. Saves updated models to the `models/` directory
    4. Reloads the KNN model for immediate use
    
    **Note:** This operation may take several minutes depending on dataset size.
    
    **Returns:**
    - `status`: "success" or "error"
    - `message`: Detailed information about the retraining process
    """
    try:
        print("Starting model retraining...")
        results_df = train_all_models(
            csv_path='alzheimers_disease_data.csv',
            results_dir='results',
            models_dir='models'
        )
        
        # reload the best model and scaler
        if load_model():
            return RetrainOutput(
                status="success",
                message=f"All models retrained and best model ({model_name if model_name else 'Best Model'}) reloaded successfully"
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


@app.get("/health", tags=["Monitoring"])
async def health():
    """
    Health check endpoint for API and model status.
    
    Returns the current status of the API, including whether the model
    and scaler are properly loaded and ready for predictions.
    
    **Response Fields:**
    - `status`: Overall API health status
    - `model_loaded`: Boolean indicating if the KNN model is loaded
    - `model_path`: Path to the model file
    - `model_type`: Type of model being used
    - `scaler_loaded`: Boolean indicating if the feature scaler is loaded
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_type": model_name if model_name else "Unknown",
        "scaler_loaded": scaler is not None
    }
