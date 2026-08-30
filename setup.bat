@echo off
echo ===================================================
echo Setting up ALS AI Screening Platform
echo ===================================================

echo [1/3] Setting up Python Virtual Environment...
cd backend
python -m venv .venv
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..

echo [2/3] Setting up Environment File...
if not exist .env (
    copy .env.example .env
    echo Created .env from .env.example
) else (
    echo .env already exists.
)

echo [3/3] Installing Frontend Dependencies...
cd frontend
npm install
cd ..

echo ===================================================
echo Setup complete! Run "run.bat" to start the servers.
echo ===================================================
pause
