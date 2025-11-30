# Test script for Docker setup (PowerShell)
# Usage: .\test_docker.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AEyePro Docker Test Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if Docker Compose is installed
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker Compose is not installed. Please install Docker Compose first." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker and Docker Compose are installed" -ForegroundColor Green

# Check if .env file exists
if (-not (Test-Path "src/chatbot/.env")) {
    Write-Host "⚠️  Warning: src/chatbot/.env not found. Please create it with GOOGLE_API_KEY." -ForegroundColor Yellow
    Write-Host "   Container may fail to start without API key." -ForegroundColor Yellow
} else {
    Write-Host "✅ .env file found" -ForegroundColor Green
}

# Check if data directory exists
if (-not (Test-Path "src/data")) {
    Write-Host "⚠️  Warning: src/data directory not found. Creating it..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "src/data" -Force | Out-Null
    Write-Host "✅ Created src/data directory" -ForegroundColor Green
} else {
    Write-Host "✅ Data directory exists" -ForegroundColor Green
}

# Build and test
Write-Host ""
Write-Host "Building Docker image..." -ForegroundColor Cyan
docker-compose build

Write-Host ""
Write-Host "Starting container..." -ForegroundColor Cyan
docker-compose up -d

Write-Host ""
Write-Host "Waiting for container to be healthy (max 60 seconds)..." -ForegroundColor Cyan
$timeout = 60
$elapsed = 0
$healthy = $false

while ($elapsed -lt $timeout) {
    $status = docker-compose ps 2>&1
    if ($status -match "healthy") {
        Write-Host "✅ Container is healthy!" -ForegroundColor Green
        $healthy = $true
        break
    }
    Start-Sleep -Seconds 2
    $elapsed += 2
    Write-Host "   Waiting... (${elapsed}s/${timeout}s)" -ForegroundColor Gray
}

if (-not $healthy) {
    Write-Host "⚠️  Container did not become healthy within ${timeout} seconds" -ForegroundColor Yellow
    Write-Host "   Check logs: docker-compose logs backend" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Testing HTTP endpoint..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    Write-Host "✅ HTTP endpoint is responding" -ForegroundColor Green
} catch {
    Write-Host "❌ HTTP endpoint is not responding" -ForegroundColor Red
    Write-Host "   Check logs: docker-compose logs backend" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Container status:"
docker-compose ps

Write-Host ""
Write-Host "Recent logs:" -ForegroundColor Cyan
docker-compose logs --tail=20 backend

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  View logs:    docker-compose logs -f backend"
Write-Host "  Stop:         docker-compose down"
Write-Host "  Restart:      docker-compose restart backend"
Write-Host "==========================================" -ForegroundColor Cyan

