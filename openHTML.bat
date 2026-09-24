@echo off
cd /d "%~dp0"
start "" http://localhost:8000/static/index.html
python -m uvicorn app:app --host 127.0.0.1 --port 8000
