@setlocal enableextensions
@cd /d "%~dp0"
@echo off

set EXE=..\bin\BarcodeReaderCLI
set OUTDIR=%TEMP%\brcli
set INPDIR=.\images

REM ===== Use configuration file to setup complex processing. Demonstrates:
REM   - Reading files with language-specfic file names
REM   - Reading language-specfic barcode test values
REM   - Template-based format of TXT output 
%EXE% %OPT% %AUTH% @"encoding.config" -output=console

REM echo =================== Windows console output incorrectly represents UTF-8 text
REM echo See output files in %OUTDIR%\utf.*
REM echo Or use Windows Terminal (https://docs.microsoft.com/en-us/windows/terminal/)
