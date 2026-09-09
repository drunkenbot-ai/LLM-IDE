@echo off
rem =============================================================================
rem Standalone Cluster Worker Launcher (Local SGD)
rem =============================================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set WORKER_PY=%SCRIPT_DIR%cluster_worker.py
set WORKER_BIN_DIR=%LOCALAPPDATA%\cluster_worker
set WORKER_EXE=%WORKER_BIN_DIR%\cluster_worker.exe

rem Ensure directory exists
if not exist "%WORKER_BIN_DIR%" (
    mkdir "%WORKER_BIN_DIR%" >nul 2>&1
)

rem Create named cluster_worker.exe if it doesn't exist so Task Manager displays cluster_worker.exe
if not exist "%WORKER_EXE%" (
    where python >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)"') do (
            copy /Y "%%I" "%WORKER_EXE%" >nul 2>&1
        )
    )
)

if exist "%WORKER_EXE%" (
    set RUNNER="%WORKER_EXE%"
) else (
    set RUNNER=python
)

rem If stopped flag passed, run stop command
if "%1"=="--stop" (
    %RUNNER% "%WORKER_PY%" --stop
    goto :eof
)

if "%1"=="--status" (
    %RUNNER% "%WORKER_PY%" --status
    goto :eof
)

rem Check LLM_SHARED_DIR or arguments
if "%LLM_SHARED_DIR%"=="" (
    if "%1"=="" (
        echo [ClusterWorker] Environment variable LLM_SHARED_DIR is not set.
        echo [ClusterWorker] Please either:
        echo   1. set LLM_SHARED_DIR=\\nas\shared_folder   ^(or a local directory like C:\llm_cluster^)
        echo   2. Run: cluster_worker.bat --shared-dir C:\llm_cluster
        echo.
    )
)

%RUNNER% "%WORKER_PY%" %*
