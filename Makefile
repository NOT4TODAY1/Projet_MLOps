.PHONY: install prepare train clean help run runall lint format security ci api

install:
	python -m pip install -r requirements.txt

prepare:
	python main.py --prepare

train:
	python main.py --train

runall:
	python main.py --runall

api:
	python -m uvicorn app:app --reload

lint:
	python -m flake8 src/ main.py model_pipeline.py app.py --max-line-length=120

format:
	python -m black src/ main.py model_pipeline.py app.py --line-length=120

security:
	python -m bandit -r src/ main.py model_pipeline.py app.py

ci: lint security
	@echo "CI checks passed!"

clean:
	if exist models rmdir /s /q models
	if exist results rmdir /s /q results
	for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

help:
	@echo "install - pip install -r requirements.txt"
	@echo "prepare - python main.py --prepare"
	@echo "train - python main.py --train"
	@echo "runall - python main.py --runall"
	@echo "api - start FastAPI server (http://localhost:8000)"
	@echo "lint - flake8 linting"
	@echo "format - black code formatting"
	@echo "security - bandit security scan"
	@echo "ci - run lint + security checks"
	@echo "clean - remove models, results, caches"

run:
	python main.py
