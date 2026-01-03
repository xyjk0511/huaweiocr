@echo off
setlocal

if exist .env (
  for /f "usebackq tokens=1,* delims==" %%A in (`findstr /r /v "^\s*#"` .env) do (
    if not "%%A"=="" set "%%A=%%B"
  )
)

if "%API_KEY%"=="" (
  echo API_KEY is not set. Add it to .env or set an environment variable.
  pause
  exit /b 1
)

python run_all.py
pause
