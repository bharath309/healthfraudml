@echo off
rem NOTE: not yet verified on a physical Windows machine — test before
rem distributing to partners.
setlocal
title HealthFraudML - Bill Audit setup
echo =============================================
echo   HealthFraudML - Bill Audit setup
echo =============================================

rem --- 1. find (or install) Python --------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python isn't installed yet - downloading the official installer...
  curl -fsSL -o "%TEMP%\python_installer.exe" https://www.python.org/ftp/python/3.12.6/python-3.12.6-amd64.exe || goto :fail
  echo Installing Python quietly - takes 2-3 minutes, please wait...
  start /wait "" "%TEMP%\python_installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
  set "PY=py"
  where py >nul 2>nul || goto :fail
)
echo [1/4] Found Python:
%PY% --version

rem --- 2. install the tool ----------------------------------------------
echo [2/4] Installing HealthFraudML (may take a minute)...
%PY% -m pip install --quiet --upgrade healthfraudml || goto :fail

rem --- 3. BillAudit folder + files ----------------------------------------
set "FOLDER=%USERPROFILE%\BillAudit"
if not exist "%FOLDER%" mkdir "%FOLDER%"
echo [3/4] Creating %FOLDER% ...
curl -fsSL -o "%FOLDER%\pilot_audit.py" https://raw.githubusercontent.com/bharath309/healthfraudml/main/examples/pilot_audit.py || goto :fail
curl -fsSL -o "%FOLDER%\sample_claims_pilot.csv" https://raw.githubusercontent.com/bharath309/healthfraudml/main/examples/sample_claims_pilot.csv || goto :fail

rem --- write the everyday runner (drop CSVs in folder, double-click this) --
(
echo @echo off
echo cd /d "%%~dp0"
echo set "found="
echo for %%%%f in ^(*.csv^) do ^(
echo   set "found=1"
echo   echo Auditing %%%%f ...
echo   py pilot_audit.py "%%%%f" --provider "%%%%~nf" --out "%%%%~nf_report"
echo ^)
echo if not defined found echo No CSV files here yet - put a bill CSV in this folder first.
echo start .
echo pause
) > "%FOLDER%\run_audit.bat"

rem --- 4. verification run ------------------------------------------------
echo [4/4] Verification: auditing the sample bill...
cd /d "%FOLDER%"
%PY% pilot_audit.py sample_claims_pilot.csv --provider "Setup Test" --out setup_test >nul 2>nul
if exist "%FOLDER%\setup_test.md" (
  echo.
  echo ==============================================
  echo   SUCCESS - everything works.
  echo   Your folder: %FOLDER%
  echo   Daily use: save a bill as CSV into that
  echo   folder, then double-click run_audit.bat
  echo ==============================================
  start "" "%FOLDER%"
  pause
  exit /b 0
)

:fail
echo.
echo !! Something went wrong - open an issue: https://github.com/bharath309/healthfraudml/issues
pause
exit /b 1
