.PHONY: install prepare train runall api lint format security ci clean mlflow elk stop help

PYTHON=python3

# -----------------------
# Python tasks
# -----------------------

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) main.py --prepare

train:
	$(PYTHON) train.py

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
	rm -rf models results mlruns
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# -----------------------
# MLflow
# -----------------------

mlflow:
	mlflow server \
	--backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlruns \
	--host 0.0.0.0 \
	--port 5000

# -----------------------
# ELK Stack (Docker Compose v2)
# -----------------------

elk:
	docker compose up -d

stop:
	docker compose down

# -----------------------
# Help
# -----------------------

help:
	@echo "make install        - Install dependencies"
	@echo "make elk            - Start Elasticsearch + Kibana"
	@echo "make mlflow         - Start MLflow server"
	@echo "make train          - Train model + send logs to Elasticsearch"
	@echo "make stop           - Stop ELK stack"
