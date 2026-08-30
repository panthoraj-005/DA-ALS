#!/usr/bin/env bash
set -e

echo "==================================================="
echo "Setting up ALS AI Screening Platform"
echo "==================================================="

echo "[1/3] Setting up Python Virtual Environment..."
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..

echo "[2/3] Setting up Environment File..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
else
    echo ".env already exists."
fi

echo "[3/3] Installing Frontend Dependencies..."
cd frontend
npm install
cd ..

echo "==================================================="
echo "Setup complete! Run ./run.sh to start the servers."
echo "==================================================="
