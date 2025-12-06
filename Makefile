.PHONY: install prepare train clean help run runall lint format security ci api mlflow mlflow-sqlite

# Detect Python command (python3 for Linux/WSL, python for Windows)
PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null || echo python3)

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) main.py --prepare

train:
	$(PYTHON) main.py --train	

runall:
	$(PYTHON) main.py --runall

api:
	$(PYTHON) -m uvicorn app:app --reload

lint:
	$(PYTHON) -m flake8 src/ main.py model_pipeline.py app.py --max-line-length=120

format:
	$(PYTHON) -m black src/ main.py model_pipeline.py app.py --line-length=120

security:
	$(PYTHON) -m bandit -r src/ main.py model_pipeline.py app.py

ci: lint security
	@echo "CI checks passed!"

clean:
	@if [ -d models ]; then rm -rf models; fi
	@if [ -d results ]; then rm -rf results; fi
	@if [ -d mlruns ]; then rm -rf mlruns; fi
	@find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo "install - pip install -r requirements.txt"
	@echo "prepare - python main.py --prepare"
	@echo "train - python main.py --train"
	@echo "runall - python main.py --runall"
	@echo "api - start FastAPI server (http://localhost:8000)"
	@echo "mlflow - start MLflow UI (http://localhost:5000)"
	@echo "mlflow-sqlite - start MLflow UI with SQLite backend (http://localhost:5000)"
	@echo "lint - flake8 linting"
	@echo "format - black code formatting"
	@echo "security - bandit security scan"
	@echo "ci - run lint + security checks"
	@echo "clean - remove models, results, caches"

run:
	$(PYTHON) main.py

mlflow:
	@echo "Starting MLflow UI on http://localhost:5000"
	mlflow ui --host 127.0.0.1 --port 5000

mlflow-sqlite:
	@echo "Starting MLflow UI with SQLite backend on http://localhost:5000"
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
