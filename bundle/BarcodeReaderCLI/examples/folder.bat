@setlocal enableextensions
@cd /d "%~dp0"
@echo off

set EXE=..\bin\BarcodeReaderCLI
set OUTDIR=%TEMP%\brcli
set INPDIR=.\images

REM ===== Read barcode in a folder and sub-folders.  Linit file types to BMP and PDF
%EXE% %OPT% %AUTH% -type=pdf417 -sub -incl="*.bmp *.pdf" "%INPDIR%\"

echo =================== Windows Command windows incorrectly represents UTF-8 text
echo Use -output options to get correct UTF-8 text
echo Or use Windows Terminal (https://docs.microsoft.com/en-us/windows/terminal/)

