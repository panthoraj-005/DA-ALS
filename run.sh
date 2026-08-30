#!/usr/bin/env bash
set -e

echo "==================================================="
echo "Starting ALS AI Screening Platform"
echo "==================================================="

# Start backend in background
(cd backend && source .venv/bin/activate && python -m uvicorn main:app --port 8000 --host 127.0.0.1 --reload) &
BACKEND_PID=$!

sleep 3

# Start frontend
(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo "==================================================="
echo "Backend PID: $BACKEND_PID (http://127.0.0.1:8000)"
echo "Frontend PID: $FRONTEND_PID (http://localhost:5173)"
echo "Press Ctrl+C to terminate all services."
echo "==================================================="

trap "kill $BACKEND_PID $FRONTEND_PID" SIGINT SIGTERM
wait
