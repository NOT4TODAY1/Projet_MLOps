# Makefile pour WSL / Linux

# Chemins vers le venv
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# --- INSTALLER LES DEPENDANCES ---
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

# --- PREPARER LES DONNEES ---
prepare:
	$(PYTHON) main.py --prepare

# --- ENTRAINER LE MODELE ---
train:
	$(PYTHON) main.py --train

# --- LANCER LES TESTS ---
test:
	$(PYTHON) -m pytest tests/

# --- EXECUTER TOUT LE PIPELINE ---
run: install prepare train test

# --- CLEAN ---
clean:
	rm -rf results models __pycache__