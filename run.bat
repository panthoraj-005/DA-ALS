@echo off
echo ===================================================
echo Starting ALS AI Screening Platform
echo ===================================================

start "ALS Screening Backend (FastAPI)" cmd /k "cd backend && .venv\Scripts\activate && python -m uvicorn main:app --port 8000 --host 127.0.0.1 --reload"

timeout /t 3 /nobreak >nul

start "ALS Screening Frontend (Vite)" cmd /k "cd frontend && npm run dev"

echo ===================================================
echo Services starting!
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:5173
echo ===================================================
