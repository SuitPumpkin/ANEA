@echo off
setlocal
set "ROOT=D:\Mis Programas\ANEA"
set "PY311=C:\Python311\python.exe"
set "VENV=%ROOT%\.venv311"
set "BT=C:\BuildTools"
set "VCVARS=%BT%\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%PY311%" goto :no_py
if not exist "%VCVARS%" goto :no_vc

if exist "%VENV%\Scripts\python.exe" goto :venv_ok
echo [INFO] Creando venv .venv311 ...
"%PY311%" -m venv "%VENV%"
if errorlevel 1 goto :venv_failed

:venv_ok
set "VP=%VENV%\Scripts\python.exe"
call "%VCVARS%" >nul
if errorlevel 1 goto :vcvars_failed

echo [INFO] Actualizando pip ...
"%VP%" -m pip install --upgrade pip

echo [INFO] Instalando dependencias GUI/audio/whisper/wakeword ...
"%VP%" -m pip install --no-cache-dir customtkinter numpy pillow sounddevice soundfile faster-whisper openwakeword

echo [INFO] Instalando llama-cpp-python (wheel CPU precompilado) ...
"%VP%" -m pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

echo [INFO] Instalando TTS (build desde fuente con MSVC) ...
"%VP%" -m pip install --no-cache-dir TTS

echo [OK] Todo instalado. Ejecuta: anea.bat
goto :eof

:no_py
echo [ERROR] No se encontro %PY311%. Instala Python 3.11 desde python.org.
exit /b 1

:no_vc
echo [ERROR] No se encontro %VCVARS%. Instala MSVC Build Tools.
exit /b 1

:venv_failed
echo [ERROR] No se pudo crear el venv.
exit /b 1

:vcvars_failed
echo [ERROR] No se pudo cargar vcvars64.bat.
exit /b 1