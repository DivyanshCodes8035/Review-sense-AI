@echo off
setlocal

rem Always run from this file's project folder, even when launched by double-click.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ReviewSense AI virtual environment was not found.
    echo Expected: %~dp0.venv\Scripts\python.exe
    pause
    exit /b 1
)

echo Starting the ReviewSense AI backend and frontend...
start "ReviewSense Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn Backend.main:app --reload"
start "ReviewSense Frontend" cmd /k ".venv\Scripts\python.exe -m http.server 5500 --directory Frontend"

echo Waiting for the local servers to start...
timeout /t 5 /nobreak >nul
start "" "http://127.0.0.1:5500"

echo The website is open. Keep both server windows running while you use it.
endlocal
