@echo off
setlocal

REM Set paths and versions
set PYTHON_INSTALL_PATH=C:\Python310
set PYTHON_EXE=%PYTHON_INSTALL_PATH%\python.exe
set PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
set VENV_DIR=venv
set REQUIREMENTS=requirements.txt

REM Step 1: Download and install Python 3.10 if not installed
if not exist "%PYTHON_EXE%" (
    echo Downloading Python 3.10 installer...
    curl -o python310.exe %PYTHON_INSTALLER_URL%

    echo Installing Python 3.10...
    python310.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 TargetDir=%PYTHON_INSTALL_PATH%
    
    del python310.exe
) else (
    echo Python 3.10 already installed.
)

REM Step 2: Create virtual environment
echo Creating virtual environment...
"%PYTHON_EXE%" -m venv %VENV_DIR%

REM Step 3: Activate virtual environment and install requirements
call %VENV_DIR%\Scripts\activate.bat

echo Upgrading pip...
pip install --upgrade pip

if exist "%REQUIREMENTS%" (
    echo Installing from requirements.txt...
    pip install -r %REQUIREMENTS%
) else (
    echo ERROR: requirements.txt not found.
)

echo.
echo ✅ Setup complete. To activate the virtual environment later, run:
echo call %VENV_DIR%\Scripts\activate.bat

endlocal
pause
