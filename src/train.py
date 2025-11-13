import os
from typing import Dict, Any
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import joblib


def train_and_evaluate(models: Dict[str, Any], param_grids: Dict[str, dict], X_train, y_train, X_test, y_test, results_dir: str = 'results', models_dir: str = 'models'):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([('model', model)])
        grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='f1')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({'Model': name, 'Accuracy': acc, 'F1-score': f1})

        # Save the trained model
        model_path = os.path.join(models_dir, f"{name.replace(' ', '_')}.joblib")
        joblib.dump(best_model, model_path)
        print(f"Saved {name} to {model_path}")

    results_df = pd.DataFrame(results).sort_values(by='F1-score', ascending=False)
    results_csv = os.path.join(results_dir, 'results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"Wrote results to {results_csv}")
    return results_df
