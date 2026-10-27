@echo off
cd /d "%~dp0"
echo Downloading public email datasets and training the local classifier...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m app.training.train_email_classifier %*
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -m app.training.train_email_classifier %*
) else (
    python -m app.training.train_email_classifier %*
)
if errorlevel 1 (
    echo Training failed. Review the error above.
    pause
    exit /b 1
)
echo Training finished. You can now run start.bat.
pause
