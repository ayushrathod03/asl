@echo off
title ASL Sign Predictor & Word Autocompleter
cd /d "%~dp0"

echo ====================================================================
echo  Starting ASL Word Autocompleter & Sign Predictor Web Application...
echo ====================================================================
echo.
echo Opening browser at http://127.0.0.1:5000 in 2 seconds...
start "" timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000

python web_app.py
pause
