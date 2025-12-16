# Script PowerShell pour arrêter le conteneur Docker (Windows)

param(
    [string]$ContainerName = "fastapi-mlflow-app"
)

Write-Host "Arret du conteneur: $ContainerName" -ForegroundColor Yellow

docker stop $ContainerName 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Conteneur arrete" -ForegroundColor Green
} else {
    Write-Host "[INFO] Conteneur deja arrete ou inexistant" -ForegroundColor Yellow
}

docker rm $ContainerName 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Conteneur supprime" -ForegroundColor Green
} else {
    Write-Host "[INFO] Conteneur deja supprime" -ForegroundColor Yellow
}

