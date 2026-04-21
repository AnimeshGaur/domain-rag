#!/bin/bash
set -e

echo "=== RAGbase Setup Script ==="

echo "1. Checking/Fixing Homebrew Permissions..."
if [ -d "/opt/homebrew" ]; then
    echo "Fixing homebrew permissions (might prompt for your password):"
    sudo chown -R $(whoami) /opt/homebrew
    sudo chmod -R u+w /opt/homebrew
    export PATH="/opt/homebrew/bin:$PATH"
fi

echo "2. Installing Node.js & npm..."
if ! command -v npm &> /dev/null; then
    brew install node
else
    echo "Node.js is already installed."
fi

echo "3. Installing Docker Desktop..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker via Homebrew (this will download ~600MB)..."
    brew install --cask docker
    echo "Opening Docker. Please accept the terms and install its networking components if prompted."
    open -a Docker
    echo "Waiting for Docker daemon to start (this can take up to 60 seconds)..."
    until docker info &> /dev/null; do
        sleep 5
        echo -n "."
    done
    echo " Docker is ready!"
else
    echo "Docker is already installed."
    # Make sure it's running
    if ! docker info &> /dev/null; then
        echo "Starting Docker app..."
        open -a Docker
        until docker info &> /dev/null; do
            sleep 5
            echo -n "."
        done
        echo " Docker is ready!"
    fi
fi

echo "4. Setting up Python Environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "5. Starting Elasticsearch..."
docker-compose up elasticsearch -d

echo "6. Starting React Frontend..."
cd frontend
npm install
npm run dev &
FRONTEND_PID=$!
cd ..

echo "7. Starting FastAPI Backend..."
uvicorn main:app --reload --port 8000

# Cleanup on exit
trap "kill $FRONTEND_PID" EXIT
