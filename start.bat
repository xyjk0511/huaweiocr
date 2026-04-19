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

if not exist new_images mkdir new_images

dir /b new_images\*.jpg new_images\*.jpeg new_images\*.png new_images\*.bmp new_images\*.webp >nul 2>nul
if errorlevel 1 (
  echo Put image files into the new_images folder, then run start.bat again.
  pause
  exit /b 2
)

python run_all.py --input new_images --out runs --pause
pause
