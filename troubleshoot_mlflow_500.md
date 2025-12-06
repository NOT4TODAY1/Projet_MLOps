# Troubleshooting MLflow 500 Internal Server Error

## Causes Possibles

### 1. Runs Corrompus
Les runs MLflow peuvent être corrompus si:
- Créés avec une ancienne version de MLflow
- Interrompus pendant le training
- Fichiers `meta.yaml` incomplets

### 2. Problèmes avec les Plots
- Fichiers plots manquants ou corrompus
- Erreurs lors de la création des plots
- Problèmes de permissions

### 3. Problèmes de Permissions
- Pas de permissions d'écriture dans `mlruns/`
- Problèmes avec les fichiers Windows/WSL

### 4. Conflit de Versions
- Incompatibilité entre versions de MLflow
- Problèmes avec les dépendances

## Solutions

### Solution 1: Nettoyer les Runs Corrompus (Recommandé)

```bash
# Arrêter MLflow (Ctrl+C)

# Nettoyer complètement
python fix_mlflow_runs.py --all

# OU
rm -rf mlruns/

# Relancer le training
python main.py --train

# Relancer MLflow
make mlflow
```

### Solution 2: Vérifier les Logs

Regardez le terminal où MLflow tourne pour voir l'erreur exacte:
- Cherchez les lignes avec "ERROR" ou "Exception"
- Notez le message d'erreur complet

### Solution 3: Vérifier les Plots

Si l'erreur vient des plots:

```bash
# Vérifier si les plots existent
ls -la results/plots/

# Si problème, supprimer et retrain
rm -rf results/plots/
python main.py --train
```

### Solution 4: Redémarrer MLflow avec SQLite

```bash
# Arrêter MLflow
# Puis relancer avec SQLite backend
make mlflow-sqlite
```

### Solution 5: Vérifier les Permissions (WSL)

```bash
# Vérifier les permissions
ls -la mlruns/

# Si nécessaire, corriger
chmod -R 755 mlruns/
```

## Diagnostic Rapide

```bash
# 1. Vérifier l'état de mlruns/
ls -la mlruns/

# 2. Vérifier les logs MLflow
# (dans le terminal où MLflow tourne)

# 3. Tester avec un nouveau run simple
python demo_mlflow.py --mode basic

# 4. Si ça marche, le problème vient des runs existants
# Si ça ne marche pas, le problème vient de MLflow lui-même
```

## Solution Définitive

Si rien ne fonctionne:

```bash
# 1. Arrêter MLflow
# Ctrl+C

# 2. Nettoyer complètement
rm -rf mlruns/
rm -rf results/plots/

# 3. Relancer le training
python main.py --train

# 4. Relancer MLflow
make mlflow
```

## Prévention

Pour éviter les erreurs 500 à l'avenir:
- Toujours arrêter proprement le training (pas de Ctrl+C brutal)
- Utiliser `make clean-mlflow` avant de relancer le training
- Vérifier que les plots sont créés correctement

