# Script PowerShell pour lancer le conteneur Docker (Windows)
# Atelier 6 : Conteneurisation avec Docker

param(
    [Parameter(Mandatory=$true)]
    [string]$ImageName,
    
    [string]$ImageTag = "latest",
    [string]$ContainerName = "fastapi-mlflow-app",
    [int]$Port = 8000
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lancement du conteneur Docker" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Image: ${ImageName}:${ImageTag}" -ForegroundColor Yellow
Write-Host "Conteneur: $ContainerName" -ForegroundColor Yellow
Write-Host "Port: $Port" -ForegroundColor Yellow
Write-Host ""

# Vérifier que Docker est installé
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERREUR] Docker n'est pas installe!" -ForegroundColor Red
    exit 1
}

# Arrêter et supprimer un conteneur existant
Write-Host "Nettoyage des conteneurs existants..." -ForegroundColor Yellow
docker stop $ContainerName 2>$null
docker rm $ContainerName 2>$null

# Lancer le conteneur
Write-Host "Lancement du conteneur..." -ForegroundColor Yellow
docker run -d -p ${Port}:8000 --name $ContainerName "${ImageName}:${ImageTag}"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Conteneur demarre!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Attente de 5 secondes pour le demarrage..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # Tester l'API
    Write-Host "Test de l'API..." -ForegroundColor Yellow
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:${Port}/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "[OK] API fonctionne correctement!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Reponse:" -ForegroundColor Cyan
            $response.Content | ConvertFrom-Json | ConvertTo-Json
        }
    } catch {
        Write-Host "[ATTENTION] L'API ne repond pas encore. Verifiez les logs:" -ForegroundColor Yellow
        Write-Host "  docker logs $ContainerName" -ForegroundColor White
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Conteneur en cours d'execution" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "API disponible a: http://localhost:${Port}" -ForegroundColor Green
    Write-Host "Documentation: http://localhost:${Port}/docs" -ForegroundColor Green
    Write-Host ""
    Write-Host "Commandes utiles:" -ForegroundColor Cyan
    Write-Host "  Voir les logs: docker logs -f $ContainerName" -ForegroundColor White
    Write-Host "  Arreter: docker stop $ContainerName" -ForegroundColor White
    Write-Host "  Supprimer: docker rm $ContainerName" -ForegroundColor White
    Write-Host "  Ou utiliser: .\docker-stop-windows.ps1 -ContainerName $ContainerName" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "[ERREUR] Erreur lors du lancement du conteneur" -ForegroundColor Red
    Write-Host "Verifiez les logs: docker logs $ContainerName" -ForegroundColor Yellow
    exit 1
}

