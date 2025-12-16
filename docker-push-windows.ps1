# Script PowerShell pour pousser l'image sur Docker Hub (Windows)
# Atelier 6 : Conteneurisation avec Docker

param(
    [Parameter(Mandatory=$true)]
    [string]$DockerUser,
    
    [Parameter(Mandatory=$true)]
    [string]$ImageName,
    
    [string]$ImageTag = "latest"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Push de l'image sur Docker Hub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Utilisateur: $DockerUser" -ForegroundColor Yellow
Write-Host "Image: ${ImageName}:${ImageTag}" -ForegroundColor Yellow
Write-Host ""

# Vérifier que Docker est installé
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERREUR] Docker n'est pas installe!" -ForegroundColor Red
    exit 1
}

# Vérifier la connexion à Docker Hub
Write-Host "Verification de la connexion a Docker Hub..." -ForegroundColor Yellow
try {
    docker info | Select-String -Pattern "Username" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Connexion a Docker Hub requise..." -ForegroundColor Yellow
        docker login
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERREUR] Echec de la connexion a Docker Hub" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "Connexion a Docker Hub requise..." -ForegroundColor Yellow
    docker login
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] Echec de la connexion a Docker Hub" -ForegroundColor Red
        exit 1
    }
}

# Vérifier si l'image existe
$imageExists = docker images "${ImageName}:${ImageTag}" --format "{{.Repository}}:{{.Tag}}" 2>$null
if (-not $imageExists) {
    Write-Host "[ERREUR] L'image ${ImageName}:${ImageTag} n'existe pas localement" -ForegroundColor Red
    Write-Host "Construisez d'abord l'image:" -ForegroundColor Yellow
    Write-Host "  .\docker-build-windows.ps1 -ImageName $ImageName" -ForegroundColor White
    exit 1
}

# Tagger l'image pour Docker Hub
$hubImageName = "${DockerUser}/${ImageName}:${ImageTag}"
Write-Host "Tagging de l'image..." -ForegroundColor Yellow
docker tag "${ImageName}:${ImageTag}" $hubImageName

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Image taggee: $hubImageName" -ForegroundColor Green
} else {
    Write-Host "[ERREUR] Erreur lors du tagging" -ForegroundColor Red
    exit 1
}

# Pousser l'image
Write-Host ""
Write-Host "Push de l'image sur Docker Hub..." -ForegroundColor Yellow
Write-Host "Cela peut prendre quelques minutes..." -ForegroundColor Yellow
docker push $hubImageName

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Image poussee avec succes!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Image disponible sur Docker Hub:" -ForegroundColor Cyan
    Write-Host "  https://hub.docker.com/r/${DockerUser}/${ImageName}" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "[ERREUR] Erreur lors du push de l'image" -ForegroundColor Red
    exit 1
}

