@echo off
setlocal

set "SHOULD_PAUSE=1"
if defined CI set "SHOULD_PAUSE=0"
if /i "%HUAWEIOCR_NO_PAUSE%"=="1" set "SHOULD_PAUSE=0"
if /i "%HUAWEIOCR_NO_PAUSE%"=="true" set "SHOULD_PAUSE=0"
if /i "%HUAWEIOCR_NO_PAUSE%"=="yes" set "SHOULD_PAUSE=0"
if /i "%HUAWEIOCR_NO_PAUSE%"=="on" set "SHOULD_PAUSE=0"

set "PAUSE_ARG=--pause"
if "%SHOULD_PAUSE%"=="0" set "PAUSE_ARG="

if exist .env (
  for /f "usebackq tokens=1,* delims==" %%A in (`findstr /r /v "^\s*#"` .env) do (
    if not "%%A"=="" set "%%A=%%B"
  )
)

set "BACKEND=%CROP_INFERENCE_BACKEND%"
if "%BACKEND%"=="" set "BACKEND=local"

set "NEEDS_API_KEY=0"
if /i "%BACKEND%"=="roboflow" set "NEEDS_API_KEY=1"
if /i "%BACKEND%"=="remote" set "NEEDS_API_KEY=1"
if /i "%BACKEND%"=="cloud" set "NEEDS_API_KEY=1"

if "%NEEDS_API_KEY%"=="1" if "%API_KEY%"=="" (
  echo API_KEY is not set. Add it to .env or set an environment variable.
  call :maybe_pause
  exit /b 1
)

if not exist new_images mkdir new_images

dir /b new_images\*.jpg new_images\*.jpeg new_images\*.png new_images\*.bmp new_images\*.webp >nul 2>nul
if errorlevel 1 (
  echo Put image files into the new_images folder, then run start.bat again.
  call :maybe_pause
  exit /b 2
)

python run_all.py --input new_images --out runs %PAUSE_ARG%
set "RUN_EXIT=%ERRORLEVEL%"
call :maybe_pause
exit /b %RUN_EXIT%

:maybe_pause
if "%SHOULD_PAUSE%"=="1" pause
exit /b 0
