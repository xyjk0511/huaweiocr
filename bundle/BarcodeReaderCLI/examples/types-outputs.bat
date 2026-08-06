@setlocal enableextensions
@cd /d "%~dp0"
@echo off

set EXE=..\bin\BarcodeReaderCLI
set OUTDIR=%TEMP%\brcli
set INPDIR=.\images

REM ===== Read availbale barcode types
REM ===== Save output in JSON and CSV formats
%EXE% %OPT%  %AUTH% -type=pdf417,qr,datamatrix,code39,code128,codabar,ucc128,code93,upca,ean8,upce,ean13,i25,imb,bpo,aust,sing   "%INPDIR%\types.pdf"  -output="%OUTDIR%\types.json" -output="%OUTDIR%\types.csv" 

