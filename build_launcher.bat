@echo off
setlocal
REM Build single-file bootstrap EXE

echo ========================================
echo MeteorDetector Bootstrap Build
echo ========================================
echo.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

set PY_EXE=
if exist "%SCRIPT_DIR%\.conda\python.exe" (
    set PY_EXE=%SCRIPT_DIR%\.conda\python.exe
) else (
    set PY_EXE=python
)

echo [1/4] Python: %PY_EXE%
"%PY_EXE%" -c "import sys;print(sys.version)" || goto :error

echo [2/4] Ensure PyInstaller...
"%PY_EXE%" -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    "%PY_EXE%" -m pip install pyinstaller || goto :error
)

echo [3/4] Clean previous output...
if exist "dist\MeteorDetectorBootstrap.exe" del /f /q "dist\MeteorDetectorBootstrap.exe"
if exist "build\MeteorDetectorBootstrap" rmdir /s /q "build\MeteorDetectorBootstrap"

echo [4/4] Build onefile bootstrap...
"%PY_EXE%" -m PyInstaller Launcher.spec --noconfirm --clean || goto :error

echo.
echo ========================================
echo Build completed
echo ========================================
echo Output: dist\MeteorDetectorBootstrap.exe
echo Distribute this EXE only.
echo.
exit /b 0

:error
echo.
echo Build failed.
exit /b 1
