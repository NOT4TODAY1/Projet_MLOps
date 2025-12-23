import argparse
import os
from model_pipeline import train_all_models, prepare_data, load_model
from elasticsearch import Elasticsearch
import requests
from datetime import datetime, timezone
import psutil
from elasticsearch import Elasticsearch
from datetime import datetime

# Expose FastAPI app for ASGI servers (e.g. `uvicorn main:app`).
# This keeps the CLI behavior intact while allowing Docker/uvicorn to
# import `app` from this module.
try:
    from app import app as app  # pragma: no cover
except Exception:
    app = None

es = Elasticsearch(
    "http://localhost:9200",
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8",
    },
)

try:
    ping_ok = es.ping()
except Exception:
    ping_ok = False
print("Elasticsearch connected:", ping_ok)

def log_system_metrics():
    doc = {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "ram_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "timestamp": datetime.utcnow()
    }
    es.index(index="system-metrics", document=doc)

log_system_metrics()



def index_metrics(results_df, model_name: str = 'Random Forest'):
    # pick row
    row = results_df[results_df['Model'] == model_name]
    if row.empty:
        row = results_df.iloc[0]
        model_name = row['Model']
    else:
        row = row.iloc[0]

    log_doc = {
        "timestamp": datetime.now(timezone.utc),
        "model": model_name,
        "train_accuracy": float(row['TrainAccuracy']),
        "test_accuracy": float(row['TestAccuracy']),
        "f1_score": float(row['F1-score']),
    }

    print("📤 Sending to Elasticsearch:", log_doc)
    try:
        res = es.index(index="mlflow-metrics", document=log_doc)
        # some clients may not return a dict with 'result'
        result_text = res.get("result") if isinstance(res, dict) else str(res)
        print("✅ Indexed with result:", result_text)
    except Exception as e:
        # Do not crash the pipeline if Elasticsearch is unavailable
        print("⚠️  Elasticsearch indexing failed; continuing without indexing.")
        print("  Error:", type(e).__name__, str(e))

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default='alzheimers_disease_data.csv')
    p.add_argument('--train', action='store_true')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--runall', action='store_true')
    p.add_argument('--models-dir', type=str, default='models')
    p.add_argument('--results-dir', type=str, default='results')
    p.add_argument('--load-model', type=str)
    return p


def main():
    p = build_parser()
    args = p.parse_args()

    if args.runall:
        print('Running full pipeline...')
        print('Step 1: Preparing data...')
        X_train, X_test, y_train, y_test, _scaler = prepare_data(args.csv)
        print('X_train', getattr(X_train, 'shape', 'N/A'))
        print('X_test', getattr(X_test, 'shape', 'N/A'))
        print('y_train', getattr(y_train, 'shape', 'N/A'))
        print('y_test', getattr(y_test, 'shape', 'N/A'))
        print('Step 2: Training all models...')
        results_df = train_all_models(csv_path=args.csv, results_dir=args.results_dir, models_dir=args.models_dir)
        print('\nFinal Results:')
        print(results_df)
        print(f'\nAll files saved to models/ and results/')
        index_metrics(results_df)


    if args.prepare:
        X_train, X_test, y_train, y_test, _scaler = prepare_data(args.csv)
        print('X_train', getattr(X_train, 'shape', 'N/A'))
        print('X_test', getattr(X_test, 'shape', 'N/A'))
        print('y_train', getattr(y_train, 'shape', 'N/A'))
        print('y_test', getattr(y_test, 'shape', 'N/A'))
        return

    if args.train:
        print('Training...')
        results_df = train_all_models(csv_path=args.csv, results_dir=args.results_dir, models_dir=args.models_dir)
        print(results_df)
        try:
            index_metrics(results_df)
        except Exception:
            pass
        return

    if args.load_model:
        m = load_model(args.load_model)
        print('Loaded', m)
        return

    p.print_help()


if __name__ == '__main__':
    main()
