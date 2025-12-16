# Script PowerShell pour construire l'image Docker (Windows)
# Atelier 6 : Conteneurisation avec Docker

param(
    [Parameter(Mandatory=$true)]
    [string]$ImageName,
    
    [string]$ImageTag = "latest"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Construction de l'image Docker" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Image: ${ImageName}:${ImageTag}" -ForegroundColor Yellow
Write-Host ""

# Vérifier que Docker est installé
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERREUR] Docker n'est pas installe!" -ForegroundColor Red
    Write-Host "Installez Docker Desktop depuis: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    exit 1
}

# Vérifier que Docker fonctionne
try {
    $dockerVersion = docker --version
    Write-Host "[OK] Docker detecte: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERREUR] Docker ne fonctionne pas. Assurez-vous que Docker Desktop est demarre." -ForegroundColor Red
    exit 1
}

# Vérifier que les modèles sont entraînés
if (-not (Test-Path "models\best_model.joblib")) {
    Write-Host "[ATTENTION] Les modeles ne sont pas entraines!" -ForegroundColor Yellow
    Write-Host "Voulez-vous entrainer les modeles maintenant? (O/N)" -ForegroundColor Cyan
    $response = Read-Host
    if ($response -eq "O" -or $response -eq "o") {
        Write-Host "Entrainement des modeles..." -ForegroundColor Yellow
        python main.py --train
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERREUR] Erreur lors de l'entrainement" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "[ERREUR] Les modeles doivent etre entraines avant la construction de l'image" -ForegroundColor Red
        exit 1
    }
}

# Construire l'image
Write-Host ""
Write-Host "Construction de l'image..." -ForegroundColor Yellow
docker build -t "${ImageName}:${ImageTag}" .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Image construite avec succes!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Pour verifier l'image:" -ForegroundColor Cyan
    Write-Host "  docker images" -ForegroundColor White
    Write-Host ""
    Write-Host "Pour lancer le conteneur:" -ForegroundColor Cyan
    Write-Host "  .\docker-run-windows.ps1 -ImageName $ImageName" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "[ERREUR] Erreur lors de la construction de l'image" -ForegroundColor Red
    exit 1
}

