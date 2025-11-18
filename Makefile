
.PHONY: install prepare train clean help run runall lint format security ci

install:
	python -m pip install -r requirements.txt

prepare:
	python main.py --prepare

train:
	python main.py --train

runall:
	python main.py --runall

lint:
	flake8 src/ main.py model_pipeline.py --max-line-length=120

format:
	black src/ main.py model_pipeline.py --line-length=120

security:
	bandit -r src/ main.py model_pipeline.py

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
	@echo "lint - flake8 linting"
	@echo "format - black code formatting"
	@echo "security - bandit security scan"
	@echo "ci - run lint + security checks"
	@echo "clean - remove models, results, caches"

run:
	python main.py
