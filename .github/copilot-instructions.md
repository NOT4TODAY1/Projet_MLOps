# MLOps Alzheimer's Disease Classification - AI Agent Guide

## Project Overview

This is a **machine learning classification project** predicting Alzheimer's disease diagnosis from medical and behavioral patient data. The workflow is a complete ML pipeline: data exploration → preprocessing → model training → evaluation.

**Key Files:**
- `alzheimers_project.ipynb` - Single-file Jupyter notebook containing entire workflow
- `alzheimers_disease_data.csv` - Patient dataset with 35 features and binary diagnosis target (0=healthy, 1=Alzheimer's)

## Data Architecture & Processing Pipeline

### Data Characteristics
- **Size:** ~2,150 patient records with 37 columns (PatientID, 35 features, DoctorInCharge, Diagnosis)
- **Target:** `Diagnosis` (binary: 0 or 1)
- **Feature Types:** Demographic (Age, Gender, Education), Biometric (BMI, BP, Cholesterol), Cognitive (MMSE, ADL, FunctionalAssessment), Lifestyle (Smoking, AlcoholConsumption, PhysicalActivity), Health History (CardiovascularDisease, Diabetes, Depression)

### Data Processing Pattern (Cell-by-cell order)
1. **Load & Explore:** `pd.read_csv()` → shape, dtypes, nulls, statistics
2. **Cleaning:** Drop `PatientID`, `DoctorInCharge`; remove duplicates
3. **Standardization:** `StandardScaler()` on 15 numerical features (see cell with `columns = [...]` list)
   - **Critical:** ALL 15 listed columns must be scaled together for model consistency
4. **Train/Test Split:** `train_test_split(test_size=0.2, random_state=40, shuffle=True)`

### Feature Scaling Pattern
Always scale these 15 features as a unit before modeling:
```python
columns = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 
           'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
           'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 
           'MMSE', 'FunctionalAssessment', 'ADL']
standard_scaler = StandardScaler()
df[columns] = standard_scaler.fit_transform(df[columns])
```

## Model Training Convention

### Parameter Grid Pattern
Models use hyperparameter tuning via `GridSearchCV` with pipeline structure:
```python
param_grid = {'model__param_name': [values]}  # Note: 'model__' prefix for Pipeline
GridSearchCV(model, param_grid, cv=5, scoring='f1')
```

### Three Standard Models
1. **Decision Tree:** `max_depth: [3, 5, 7, None]`
2. **Random Forest:** `n_estimators: [50, 100, 200]`, `max_depth: [3, 5, 7, None]`
3. **K-Nearest Neighbors:** `n_neighbors: [3, 5, 7]`

All use `f1` scoring (handles imbalanced target), 5-fold CV.

### ANN Model (Specialized)
- Keras Sequential: Input → Dense(32, relu) → Dense(16, relu) → Dense(1, sigmoid)
- Optimizer: Adam(learning_rate=0.001)
- Loss: binary_crossentropy
- Training: 50 epochs, batch_size=32, validation split on test set

## Evaluation & Visualization Patterns

### Metrics Function
- `evaluate_model()` returns (train_accuracy, test_accuracy) - use for overfitting detection
- `save_model_result()` appends to global `results` list → converts to DataFrame
- Final comparison: Sort by F1-score (not accuracy)

### Standard Visualizations
- **Distribution checks:** `sns.histplot()` with KDE for all numerical columns
- **Diagnosis by metric:** `sns.swarmplot()` for cognitive scores (MMSE, ADL, FunctionalAssessment)
- **Correlation:** `sns.heatmap()` on key columns with `annot=True`
- **Prediction quality:** Custom Plotly histogram comparing actual vs predicted distributions

### French Labels Convention
Comments and plot titles use French (e.g., "Diagnostic", "Âge", "Répartition des diagnostics"). Preserve this for consistency.

## Common Extensions & Patterns

### If adding new preprocessing:
- Insert before standardization; ensure new features are added to the `columns` list
- Run through EDA visualizations before modeling

### If evaluating new models:
- Use same Pipeline + GridSearchCV pattern
- Report F1-score in final comparison (not just accuracy)
- Include train/test accuracy to detect overfitting

### If modifying neural network:
- Keep binary_crossentropy loss (binary classification)
- Validation data should be test set, not separate validation split
- Track history curves (loss and accuracy) across epochs

## Global State & Notebook Execution Order

⚠️ **Critical:** The notebook depends on cell execution order:
- `results = []` (initialized early) accumulates model results
- `results_df` DataFrame is built incrementally via `save_model_result()`
- Rerunning cells updates global state; clear `results` if starting fresh

**Execution flow:** Load data → Clean → Visualize → Scale → Split → Train each model → Evaluate → Compare final results

## Key Imports & Dependencies
```python
pandas, numpy, matplotlib, seaborn, plotly  # Data & visualization
sklearn: MinMaxScaler, StandardScaler, train_test_split, 
         DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier,
         accuracy_score, classification_report, confusion_matrix, 
         GridSearchCV, Pipeline  # ML pipeline
tensorflow.keras: Sequential, Dense, Adam  # Neural networks
```

