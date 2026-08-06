@setlocal enableextensions
@cd /d "%~dp0"
@echo off

set EXE=..\bin\BarcodeReaderCLI
set OUTDIR=%TEMP%\brcli
set INPDIR=.\images

REM ===== Read Code39 barcode.   Output is default JSON format
%EXE% %OPT% %AUTH% -type=code39 "%INPDIR%\test.tif"

REM ===== Read Code39 barcode.  Output just text value to console
%EXE% %OPT% %AUTH% -type=code39 "%INPDIR%\test.tif" -format=text --output-text="{text}"
