# Quick Start Guide

## 🚀 Running the Project

### Option 1: Using PowerShell (Recommended)

**Backend (Terminal 1):**
```powershell
cd "C:\Users\Hazem\Desktop\4DS8\MLOps\Projet1.1\Projet_MLOps"
.\venv\Scripts\Activate.ps1
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**Frontend (Terminal 2):**
```powershell
cd "C:\Users\Hazem\Desktop\4DS8\MLOps\Projet1.1\Projet_MLOps\frontend"
npm run dev
```

### Option 2: Using Makefile

```bash
# Terminal 1 - Backend
make api

# Terminal 2 - Frontend  
cd frontend && npm run dev
```

## 📍 Access Points

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## ✨ New Features Available

### 1. **Prediction Confidence Display**
- Visual progress bars showing confidence levels
- Percentage displays for both classes
- Color-coded indicators

### 2. **Export Functionality**
- **Export PDF**: Professional reports with patient data
- **Export CSV**: Spreadsheet-compatible format
- **Export JSON**: Save patient data

### 3. **Prediction History**
- Automatic saving of all predictions
- View past assessments with timestamps
- Load previous predictions back into form
- Delete individual items or clear all

### 4. **Batch Processing**
- Upload multiple JSON files at once
- Process all files automatically
- Download results as CSV

### 5. **Feature Importance**
- Shows which factors influenced the prediction
- Highlights high/medium/low impact features
- Explains why each feature matters

### 6. **Data Validation Warnings**
- Real-time validation of unusual values
- Warnings for:
  - Very low MMSE scores (< 20)
  - Abnormal blood pressure
  - Extreme BMI values
  - Multiple symptoms present
- Color-coded by severity

### 7. **Model Information Panel**
- View model status, type, and version
- Check if model and scaler are loaded
- Toggleable panel in header

### 8. **Theme Toggle**
- Dark/Light mode switch
- Preference saved automatically

### 9. **Help Tooltips**
- Info icons next to form fields
- Hover to see explanations
- Contextual help for medical terms

### 10. **Quick Actions**
- **Copy to Clipboard**: Copy prediction results
- **Keyboard Shortcuts**:
  - `Ctrl/Cmd + Enter`: Submit form
  - `Ctrl/Cmd + S`: Export patient data as JSON
- **Auto-save**: Form data saved automatically

## 🎯 Quick Test

1. Open http://localhost:3000
2. Click "Load Sick Patient Example"
3. Click "Predict"
4. View the confidence scores and feature importance
5. Try exporting as PDF or CSV
6. Check the History panel

## 📝 Notes

- Make sure both servers are running before using the app
- The backend needs the models trained (run `python main.py --train` first)
- All predictions are automatically saved to browser localStorage
- Theme preference is saved automatically

