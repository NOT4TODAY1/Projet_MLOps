# Projet_MLOps — Alzheimer's Disease Classification

**Overview**

This repository contains a modularized machine learning pipeline that reproduces a Jupyter notebook workflow for classifying Alzheimer's disease using patient data. The project includes data preparation, model training (with hyperparameter tuning), evaluation and a FastAPI service to expose predictions.

**Contents**

- `alzheimers_disease_data.csv` — source dataset (CSV).
- `alzheimers_project.ipynb` — original notebook (reference).
- `main.py` — CLI entrypoint to run prepare/train/runall/load operations.
- `model_pipeline.py` — modular pipeline functions: `prepare_data`, `train_model`, `evaluate_model`, `save_model`, `load_model`, `train_all_models`.
- `app.py` — FastAPI application exposing `/predict`, `/retrain`, `/health`.
- `api_client.py` — small test client for the API.
- `Makefile` — convenience targets (install, prepare, train, runall, api, mlflow, mlflow-sqlite, lint, format, security, ci, clean).
- `requirements.txt` — Python dependencies.
- `models/` — trained model files (joblib) (generated at training time).
- `results/` — results CSV created after training.
- `src/` — small helper modules (`data.py`, `preprocess.py`, `models.py`).

**Quick start (recommended)**

### Backend Setup

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**WSL (Linux):**
```bash
python3 -m venv venv_wsl
source venv_wsl/bin/activate
pip install -r requirements.txt
```

📖 **For WSL setup details, see [WSL_SETUP.md](WSL_SETUP.md)**

2. Train all models (this runs GridSearchCV and may take several minutes):

```powershell
python main.py --train
# or
python main.py --runall   # prepares and then trains
```

Trained models are saved to the `models/` directory and results CSV is saved in `results/results.csv`.

3. Start the backend API server:

```powershell
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at http://localhost:8000

### Frontend Setup

1. Navigate to frontend directory and install dependencies:

```powershell
cd frontend
npm install
```

2. Start the development server:

```powershell
npm run dev
```

The frontend will be available at http://localhost:3000

### Running Both Servers

**Terminal 1 - Backend:**
```powershell
cd "C:\Users\Hazem\Desktop\4DS8\MLOps\Projet1.1\Projet_MLOps"
.\venv\Scripts\Activate.ps1
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend:**
```powershell
cd "C:\Users\Hazem\Desktop\4DS8\MLOps\Projet1.1\Projet_MLOps\frontend"
npm run dev
```

Then open **http://localhost:3000** in your browser!

📖 **For detailed instructions, see [HOW_TO_RUN.md](HOW_TO_RUN.md)**

**MLflow Integration (Atelier 5)**

The project includes MLflow for experiment tracking. All training runs are automatically logged with:
- **Parameters** (`mlflow.log_param()`): Dataset info, hyperparameters, best parameters from GridSearchCV
- **Metrics** (`mlflow.log_metric()`): Train/test accuracy, F1-score, precision, recall, confusion matrix components
- **Models** (`mlflow.sklearn.log_model()`): All trained models in MLflow format
- **Artifacts**: Model files (.joblib), results CSV, and **visualization plots**
  - Confusion matrix plots (per model)
  - ROC curves (per model)
  - Metrics comparison chart (all models)

To view experiments in MLflow UI:

**Windows (PowerShell):**
```powershell
# Direct command (make not available on Windows)
.\venv\Scripts\Activate.ps1
mlflow ui --host 127.0.0.1 --port 5000

# With SQLite backend
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

**WSL/Linux:**
```bash
# With Makefile (recommended)
source venv_wsl/bin/activate
make mlflow

# Or directly
mlflow ui --host 0.0.0.0 --port 5000
```

**Note:** On Windows, use `--host 127.0.0.1`. On WSL, use `--host 0.0.0.0` to allow access from Windows.

Then open http://localhost:5000 in your browser to view:
- All training experiments
- Model comparisons
- Hyperparameter tuning results
- Model artifacts
- **Visualization plots** (in Artifacts → plots/)

**MLflow Workflow:**

1. **Train models** (automatically logs to MLflow):
   ```bash
   python main.py --train
   ```

2. **View results in MLflow UI**:
   ```bash
   make mlflow  # or mlflow ui --host 0.0.0.0 --port 5000
   ```

3. **Access plots**: Click on any run → Artifacts → plots/ to see:
   - Confusion matrices
   - ROC curves
   - Metrics comparison

**Troubleshooting:** If you see errors about corrupted runs (`TypeError: __init__() missing 1 required positional argument: 'run_uuid'`) or 500 Internal Server Error, clean the MLflow data:

```bash
# Option 1: Use Makefile (recommended)
make clean-mlflow
make train

# Option 2: Manual cleanup
rm -rf mlruns/
python main.py --train
```

For detailed troubleshooting, see `troubleshoot_mlflow_500.md`.

**Run the FastAPI prediction service**

Start the API server (development mode):

```powershell
python -m uvicorn app:app --reload
```

Open the auto docs in your browser:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Predict (example)**

Send a POST request to `/predict` with a JSON body containing the required features. Example (PowerShell):

```powershell
$body = @{
  Age = 45; Gender = 1; Ethnicity = 2; EducationLevel = 4; BMI = 23.5;
  Smoking = 0; AlcoholConsumption = 1; PhysicalActivity = 4; DietQuality = 8;
  SleepQuality = 7; FamilyHistoryAlzheimers = 0; CardiovascularDisease = 0;
  Diabetes = 0; Depression = 0; HeadInjury = 0; Hypertension = 0;
  SystolicBP = 120; DiastolicBP = 80; CholesterolTotal = 180; CholesterolLDL = 100;
  CholesterolHDL = 60; CholesterolTriglycerides = 120; MMSE = 29; FunctionalAssessment = 9;
  MemoryComplaints = 0; BehavioralProblems = 0; ADL = 6; Confusion = 0; Disorientation = 0;
  PersonalityChanges = 0; DifficultyCompletingTasks = 0; Forgetfulness = 0
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:8000/predict -Method POST -Body $body -ContentType 'application/json'
```

Or with curl:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @sample.json
```

Where `sample.json` contains the JSON payload shown above.

**Retrain endpoint (optional/excellence)**

The API exposes `/retrain` which will re-run the full training pipeline and overwrite models in `models/`. Use carefully — this can take several minutes:

```bash
curl -X POST "http://localhost:8000/retrain"
```

**Makefile targets**

- `install` - install deps from `requirements.txt`.
- `prepare` - run `main.py --prepare`.
- `train` - run `main.py --train`.
- `runall` - prepare + train.
- `api` - run FastAPI server (uses `uvicorn app:app`).
- `mlflow` - start MLflow UI on http://localhost:5000.
- `mlflow-sqlite` - start MLflow UI with SQLite backend on http://localhost:5000.
- `lint` - run `flake8`.
- `format` - run `black`.
- `security` - run `bandit`.
- `ci` - run `lint` and `security`.
- `clean` - remove generated artifacts.

**Notes & troubleshooting**

- The API prediction model expects the same feature names used in training (see dataset columns). If you get a `Feature names` mismatch error, ensure your JSON keys match dataset column names exactly.
- If you run into missing CLI `make` on Windows, use the Python CLI commands above, or install make via `choco install make` or use WSL.
- The project uses `joblib` to save models (files under `models/`); do not edit those manually.

**Next steps / improvements**

- Persist the scaler used during preprocessing and apply it in the API (currently a simple live-scaling is applied; persisting the actual fitted scaler used during training is more correct).
- Add unit tests for `prepare_data`, `train_model`, and `evaluate_model`.
- Add Dockerfile and docker-compose for API deployment.

---

If you want, I can now:
- Persist the actual scaler used during training and load it in `app.py` (recommended),
- Add a `Dockerfile` for the API,
- Add basic unit tests and a `pytest` Makefile target.

Tell me which of these you'd like next.