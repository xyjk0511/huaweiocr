@setlocal enableextensions
@cd /d "%~dp0"
@echo off

set EXE=..\bin\BarcodeReaderCLI
set OUTDIR=%TEMP%\brcli
set INPDIR=.\images

REM ===== Read barcoded from Web-based images.   
REM ===== Read multiple images with a single call, applying different 'type' option to each image 
%EXE% %OPT% %AUTH% -type=pdf417 "https://wabr.inliteresearch.com/SampleImages/drvlic.ca.jpg" -type=code39  "https://www.dropbox.com/s/qcd8zfdvckwwdem/img39.pdf?dl=1" 


