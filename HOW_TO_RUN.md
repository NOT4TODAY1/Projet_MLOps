# 🚀 How to Run the Alzheimer's Disease Classifier Project

## Prerequisites

- **Python 3.8+** installed
- **Node.js 18+** and **npm** installed
- **Git** (optional, for cloning)

## Step-by-Step Instructions

### Step 1: Navigate to Project Directory

```powershell
cd "C:\Users\Hazem\Desktop\4DS8\MLOps\Projet1.1\Projet_MLOps"
```

### Step 2: Set Up Python Backend

#### 2.1 Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2.2 Install Python Dependencies (if not already installed)

```powershell
pip install -r requirements.txt
```

#### 2.3 Train the Models (First Time Only)

```powershell
python main.py --train
```

This will:
- Train all models (Decision Tree, Random Forest, KNN, Logistic Regression, SVM, AdaBoost)
- Save the best model to `models/best_model.joblib`
- Save the scaler to `models/scaler.joblib`
- Create results in `results/results.csv`

**Note:** This takes a few minutes. You only need to do this once, or when you want to retrain.

### Step 3: Start the Backend API Server

**Open a NEW PowerShell/Terminal window:**

```powershell
cd "C:\Users\Hazem\Desktop\4DS8\MLOps\Projet1.1\Projet_MLOps"
.\venv\Scripts\Activate.ps1
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Keep this terminal window open!**

### Step 4: Set Up React Frontend

**Open ANOTHER NEW PowerShell/Terminal window:**

```powershell
cd "C:\Users\Hazem\Desktop\4DS8\MLOps\Projet1.1\Projet_MLOps\frontend"
npm install
```

This installs all frontend dependencies. Only needed once.

### Step 5: Start the Frontend Development Server

**In the same frontend terminal:**

```powershell
npm run dev
```

You should see:
```
VITE v5.x.x  ready in xxxx ms
➜  Local:   http://localhost:3000/
```

**Keep this terminal window open too!**

### Step 6: Open the Application

Open your web browser and go to:
```
http://localhost:3000
```

## 🎯 Quick Start (If Everything is Already Set Up)

If you've already installed dependencies and trained models:

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

Then open: **http://localhost:3000**

## 📋 Alternative: Using Makefile (If Available)

If you have `make` installed:

```bash
# Terminal 1 - Backend
make api

# Terminal 2 - Frontend
cd frontend && npm run dev
```

## ✅ Verify Everything is Working

1. **Backend Health Check:**
   - Open: http://localhost:8000/health
   - Should show: `{"status":"healthy","model_loaded":true,...}`

2. **API Documentation:**
   - Open: http://localhost:8000/docs
   - Should show Swagger UI with all endpoints

3. **Frontend:**
   - Open: http://localhost:3000
   - Should show the Alzheimer's Disease Classifier form

## 🧪 Test the Application

1. Click **"Load Sick Patient Example"** button
2. Click **"Predict"**
3. You should see:
   - Prediction result (Alzheimer's Disease)
   - Confidence scores with progress bars
   - Feature importance visualization
4. Try exporting as PDF or CSV
5. Check the History panel

## 🛠️ Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'mlflow'`
- **Solution:** Make sure virtual environment is activated and run `pip install -r requirements.txt`

**Problem:** `Model not found at models/best_model.joblib`
- **Solution:** Run `python main.py --train` first

**Problem:** Port 8000 already in use
- **Solution:** Change port: `python -m uvicorn app:app --reload --port 8001`

### Frontend Issues

**Problem:** `npm: command not found`
- **Solution:** Install Node.js from https://nodejs.org/

**Problem:** Port 3000 already in use
- **Solution:** Vite will automatically use the next available port (check terminal output)

**Problem:** `Cannot connect to API`
- **Solution:** Make sure backend is running on http://localhost:8000

### General Issues

**Problem:** PowerShell execution policy error
- **Solution:** Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Problem:** Models not predicting correctly
- **Solution:** Make sure models are trained: `python main.py --train`

## 📝 Important Files

- `app.py` - FastAPI backend server
- `model_pipeline.py` - Model training pipeline
- `main.py` - CLI for training models
- `frontend/src/App.tsx` - React frontend
- `models/best_model.joblib` - Trained model (generated)
- `models/scaler.joblib` - Feature scaler (generated)
- `alzheimers_disease_data.csv` - Training data

## 🎓 Next Steps

Once running:
1. Fill in patient data manually
2. Upload JSON/PDF files with patient data
3. Use batch processing for multiple patients
4. Export results as PDF/CSV
5. View prediction history
6. Explore all the new features!

## 📞 Need Help?

- Check terminal windows for error messages
- Verify both servers are running
- Check that models are trained
- Ensure ports 8000 and 3000 are available

