from typing import Tuple, Any, Dict
import os
import joblib
import pandas as pd
import shutil
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import numpy as np

from src.data import load_data, split_features_target
from src.preprocess import clean_dataframe, scale_features
from src.models import get_models_and_grids


def prepare_data(csv_path: str = 'alzheimers_disease_data.csv', test_size: float = 0.2, random_state: int = 40, target_col: str = 'Diagnosis') -> Tuple[Any, Any, Any, Any, Any]:
    df = load_data(csv_path)
    df = clean_dataframe(df)
    df, scaler = scale_features(df)

    X, y = split_features_target(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    return X_train, X_test, y_train, y_test, scaler


def train_model(model_name: str, model: Any, param_grid: Dict[str, list], X_train, y_train, cv: int = 5, scoring: str = 'f1') -> Tuple[Any, Any]:
    pipeline = Pipeline([('model', model)])
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search


def evaluate_model(model: Any, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases
        tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
        fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
        fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
        tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    return {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def save_model(model: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)


def save_results(results_df: pd.DataFrame, path: str = 'results/results.csv') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_df.to_csv(path, index=False)


def train_all_models(csv_path: str = 'alzheimers_disease_data.csv', results_dir: str = 'results', models_dir: str = 'models') -> pd.DataFrame:
    # Configure MLflow experiment (avoid apostrophe in name for compatibility)
    mlflow.set_experiment("Alzheimers Disease Classification")
    
    # Enable autologging for sklearn (optional - logs additional info automatically)
    # mlflow.sklearn.autolog()
    
    X_train, X_test, y_train, y_test, scaler = prepare_data(csv_path)
    models, param_grids = get_models_and_grids()

    results = []
    if os.path.exists(results_dir):
        try:
            shutil.rmtree(results_dir)
        except Exception:
            for f in os.listdir(results_dir):
                fp = os.path.join(results_dir, f)
                try:
                    if os.path.isdir(fp):
                        shutil.rmtree(fp)
                    else:
                        os.remove(fp)
                except Exception:
                    pass

    if os.path.exists(models_dir):
        try:
            shutil.rmtree(models_dir)
        except Exception:
            for f in os.listdir(models_dir):
                fp = os.path.join(models_dir, f)
                try:
                    if os.path.isdir(fp):
                        shutil.rmtree(fp)
                    else:
                        os.remove(fp)
                except Exception:
                    pass

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    best_model_obj = None
    best_f1 = -1.0
    best_model_name = None

    # Start MLflow parent run for the entire training session
    with mlflow.start_run(run_name="All Models Training"):
        # Log dataset information
        mlflow.log_param("dataset_path", csv_path)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("num_features", X_train.shape[1])
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Start a nested run for each model
            with mlflow.start_run(run_name=name, nested=True):
                pg = param_grids.get(name, {})
                
                # Log hyperparameters to MLflow
                # Log all hyperparameters from the grid search
                for param_name, param_values in pg.items():
                    mlflow.log_param(f"param_grid_{param_name}", str(param_values))
                
                best_model, gs = train_model(name, model, pg, X_train, y_train)
                
                # Log best hyperparameters found by GridSearchCV
                for param_name, param_value in gs.best_params_.items():
                    mlflow.log_param(f"best_{param_name}", param_value)
                
                # Log cross-validation score
                mlflow.log_metric("cv_best_score", gs.best_score_)
                
                metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)

                # Log metrics to MLflow
                mlflow.log_metric("train_accuracy", metrics['train_accuracy'])
                mlflow.log_metric("test_accuracy", metrics['test_accuracy'])
                mlflow.log_metric("test_f1", metrics['test_f1'])
                mlflow.log_metric("test_precision", metrics['test_precision'])
                mlflow.log_metric("test_recall", metrics['test_recall'])
                
                # Log confusion matrix components
                mlflow.log_metric("true_positives", metrics['true_positives'])
                mlflow.log_metric("true_negatives", metrics['true_negatives'])
                mlflow.log_metric("false_positives", metrics['false_positives'])
                mlflow.log_metric("false_negatives", metrics['false_negatives'])
                
                # Log tags for easier filtering
                mlflow.set_tag("model_type", name)
                mlflow.set_tag("experiment_type", "alzheimer_classification")
                
                # Log the model as artifact
                model_filename = f"{name.replace(' ', '_')}.joblib"
                model_path = os.path.join(models_dir, model_filename)
                save_model(best_model, model_path)
                
                # Log model artifact to MLflow
                mlflow.log_artifact(model_path, artifact_path="models")
                
                # Also log using MLflow's sklearn autologging
                mlflow.sklearn.log_model(best_model, f"sklearn_model_{name.replace(' ', '_')}")

                results.append({'Model': name, 'TrainAccuracy': metrics['train_accuracy'], 'TestAccuracy': metrics['test_accuracy'], 'F1-score': metrics['test_f1'], 'ModelPath': model_path})

                # Track best model by F1
                if metrics['test_f1'] > best_f1:
                    best_f1 = metrics['test_f1']
                    best_model_obj = best_model
                    best_model_name = name

        # Log overall best model information
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.log_param("best_model", best_model_name if best_model_name else "None")
        
        # Log results CSV as artifact
        results_df = pd.DataFrame(results).sort_values(by='F1-score', ascending=False)
        save_results(results_df, os.path.join(results_dir, 'results.csv'))
        mlflow.log_artifact(os.path.join(results_dir, 'results.csv'), artifact_path="results")

    # Persist the best model and the fitted scaler for inference
    try:
        if best_model_obj is not None:
            best_path = os.path.join(models_dir, 'best_model.joblib')
            joblib.dump(best_model_obj, best_path)
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
    except Exception as e:
        print(f"Warning: failed to persist best model or scaler: {e}")

    return results_df


if __name__ == '__main__':
    print('Training all models...')
    df = train_all_models()
    print(df)
