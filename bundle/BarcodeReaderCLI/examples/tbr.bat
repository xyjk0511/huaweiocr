@setlocal enableextensions
@cd /d "%~dp0"
@echo off

set EXE=..\bin\BarcodeReaderCLI
set OUTDIR=%TEMP%\brcli
set INPDIR=.\images

REM ===== Read barcode with TBR
%EXE% %OPT% %AUTH% -type=datamatrix -tbr=120 -fields=+tbr "%INPDIR%\dm.tbr.bmp"

