# Dockerfile pour l'application FastAPI avec MLflow
# Atelier 6 : Conteneurisation avec Docker

# Utiliser une image Python officielle comme base
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app.py .
COPY model_pipeline.py .
COPY src/ ./src/

# Copier les modèles entraînés (si disponibles)
# Note: Les modèles doivent être entraînés avant la construction de l'image
COPY models/ ./models/

# Copier les artefacts MLflow (optionnel, pour utilisation dans le conteneur)
# Note: Les runs MLflow doivent être disponibles avant la construction
COPY mlruns/ ./mlruns/

# Copier le fichier de données (nécessaire pour le retraining et la création du scaler)
COPY alzheimers_disease_data.csv .

# Exposer le port 8000 (port par défaut de FastAPI/uvicorn)
EXPOSE 8000

# Définir les variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:./mlruns

# Commande pour démarrer l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

