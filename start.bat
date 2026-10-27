@echo off
cd /d "%~dp0"
echo Activating virtual environment and starting Spam Mail Detection app...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run.py
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" run.py
) else (
    python run.py
)
pause
