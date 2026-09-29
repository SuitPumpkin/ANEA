@echo off
setlocal
set "ROOT=D:\Mis Programas\ANEA"
set "VENV=%ROOT%\.venv311"
set "BT=C:\BuildTools"
set "VCVARS=%BT%\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VCVARS%" goto :no_vcvars
if not exist "%VENV%\Scripts\python.exe" goto :no_venv

call "%VCVARS%" >nul
if errorlevel 1 goto :vcvars_failed

echo [INFO] MSVC activado. Instalando TTS...
"%VENV%\Scripts\python.exe" -m pip install --no-cache-dir TTS
if errorlevel 1 goto :pip_failed

echo [OK] TTS instalado.
goto :eof

:no_vcvars
echo [ERROR] No se encontro vcvars64.bat en %BT%
exit /b 1

:no_venv
echo [ERROR] No se encontro el venv de Python 3.11 en %VENV%
exit /b 1

:vcvars_failed
echo [ERROR] No se pudo cargar vcvars64.bat
exit /b 1

:pip_failed
echo [ERROR] pip install TTS fallo
exit /b 1