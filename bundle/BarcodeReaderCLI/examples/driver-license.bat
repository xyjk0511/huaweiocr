@setlocal enableextensions
@cd /d "%~dp0"
@echo off

set EXE=..\bin\BarcodeReaderCLI
set OUTDIR=%TEMP%\brcli
set INPDIR=.\images

REM ===== Read and decode Driver License
%EXE% %OPT% %AUTH% -type=drvlic "%INPDIR%\test.tif"


