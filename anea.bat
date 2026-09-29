@echo off
setlocal
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv311"

if not exist "%VENV%\Scripts\python.exe" (
    echo [ERROR] No se encontró el venv de Python 3.11 en %VENV%
    echo         Ejecuta scripts\install.bat primero.
    exit /b 1
)

cd /d "%ROOT%"
"%VENV%\Scripts\python.exe" -m anea.main %*