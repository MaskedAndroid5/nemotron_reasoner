@echo off
REM =============================================================================
REM run_all.bat — Windows wrapper for the Nemotron Sheaf pipeline
REM
REM Usage:
REM   run_all.bat              Run Phase 0 verification
REM   run_all.bat full         Run full pipeline
REM   run_all.bat train-only   Run verification + data + training
REM   run_all.bat submit       Package and validate existing adapter
REM
REM Requirements:
REM   - Git Bash installed (included with Git for Windows)
REM   - Python 3.10+ on PATH
REM =============================================================================

setlocal

REM Check for Git Bash
where bash >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: bash not found. Please install Git for Windows from https://git-scm.com
    echo Git Bash is included in the default installation.
    exit /b 1
)

REM Forward all arguments to the bash script
bash "%~dp0run_all.sh" %*

endlocal
