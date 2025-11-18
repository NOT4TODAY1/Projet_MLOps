from typing import Tuple, Any, Dict
import os
import joblib
import pandas as pd
import shutil
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_data, split_features_target
from src.preprocess import clean_dataframe, scale_features
from src.models import get_models_and_grids


def prepare_data(csv_path: str = 'alzheimers_disease_data.csv', test_size: float = 0.2, random_state: int = 40, target_col: str = 'Diagnosis') -> Tuple[Any, Any, Any, Any]:
    df = load_data(csv_path)
    df = clean_dataframe(df)
    df = scale_features(df)

    X, y = split_features_target(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    return X_train, X_test, y_train, y_test


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

    return {'train_accuracy': float(train_accuracy), 'test_accuracy': float(test_accuracy), 'test_f1': float(test_f1)}


def save_model(model: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)


def save_results(results_df: pd.DataFrame, path: str = 'results/results.csv') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_df.to_csv(path, index=False)


def train_all_models(csv_path: str = 'alzheimers_disease_data.csv', results_dir: str = 'results', models_dir: str = 'models') -> pd.DataFrame:
    X_train, X_test, y_train, y_test = prepare_data(csv_path)
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

    for name, model in models.items():
        print(f"Training {name}...")
        pg = param_grids.get(name, {})
        best_model, gs = train_model(name, model, pg, X_train, y_train)
        metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)

        model_filename = f"{name.replace(' ', '_')}.joblib"
        model_path = os.path.join(models_dir, model_filename)
        save_model(best_model, model_path)

        results.append({'Model': name, 'TrainAccuracy': metrics['train_accuracy'], 'TestAccuracy': metrics['test_accuracy'], 'F1-score': metrics['test_f1'], 'ModelPath': model_path})

    results_df = pd.DataFrame(results).sort_values(by='F1-score', ascending=False)
    save_results(results_df, os.path.join(results_dir, 'results.csv'))
    return results_df


if __name__ == '__main__':
    print('Training all models...')
    df = train_all_models()
    print(df)
