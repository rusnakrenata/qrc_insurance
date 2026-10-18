@echo off
REM Create a virtual environment with a supported Python and install requirements.txt (Windows).
REM The environment is created outside the repository, so OneDrive does not sync it.
setlocal
set REPO=%~dp0
set REPO=%REPO:~0,-1%
set VENV=C:\venvs\qrc_reproducibility

set PYEXE=
for %%V in (3.13 3.12 3.11) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>&1 && set PYEXE=py -%%V
    )
)
if not defined PYEXE (
    echo Python 3.11, 3.12 or 3.13 ^(64-bit^) is required.
    pause
    exit /b 1
)

if exist "%VENV%" rmdir /s /q "%VENV%"
if not exist "C:\venvs" mkdir "C:\venvs"
%PYEXE% -m venv "%VENV%" || (echo Could not create the environment. & pause & exit /b 1)
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip --quiet
"%VENV%\Scripts\python.exe" -m pip install -r "%REPO%\requirements.txt" || (echo pip install failed. & pause & exit /b 1)

echo.
echo Done. Activate with:  %VENV%\Scripts\activate
pause
