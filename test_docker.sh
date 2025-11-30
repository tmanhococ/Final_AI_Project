#!/bin/bash
# Test script for Docker setup
# Usage: ./test_docker.sh

set -e

echo "=========================================="
echo "AEyePro Docker Test Script"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Check if .env file exists
if [ ! -f "src/chatbot/.env" ]; then
    echo "⚠️  Warning: src/chatbot/.env not found. Please create it with GOOGLE_API_KEY."
    echo "   Container may fail to start without API key."
else
    echo "✅ .env file found"
fi

# Check if data directory exists
if [ ! -d "src/data" ]; then
    echo "⚠️  Warning: src/data directory not found. Creating it..."
    mkdir -p src/data
    echo "✅ Created src/data directory"
else
    echo "✅ Data directory exists"
fi

# Build and test
echo ""
echo "Building Docker image..."
docker-compose build

echo ""
echo "Starting container..."
docker-compose up -d

echo ""
echo "Waiting for container to be healthy (max 60 seconds)..."
timeout=60
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if docker-compose ps | grep -q "healthy"; then
        echo "✅ Container is healthy!"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo "   Waiting... (${elapsed}s/${timeout}s)"
done

if [ $elapsed -ge $timeout ]; then
    echo "⚠️  Container did not become healthy within ${timeout} seconds"
    echo "   Check logs: docker-compose logs backend"
fi

echo ""
echo "Testing HTTP endpoint..."
if curl -f http://localhost:5000/ > /dev/null 2>&1; then
    echo "✅ HTTP endpoint is responding"
else
    echo "❌ HTTP endpoint is not responding"
    echo "   Check logs: docker-compose logs backend"
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Container status:"
docker-compose ps

echo ""
echo "Recent logs:"
docker-compose logs --tail=20 backend

echo ""
echo "=========================================="
echo "Useful commands:"
echo "  View logs:    docker-compose logs -f backend"
echo "  Stop:         docker-compose down"
echo "  Restart:      docker-compose restart backend"
echo "=========================================="

