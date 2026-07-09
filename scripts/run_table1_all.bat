@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Run Table 1 experiments on data\benchmark\inputs.json.
REM
REM This runs the v1-style comparison:
REM   direct     = One-shot learning
REM   cot        = Zero-shot CoT
REM   multiagent = TreatAgent v2
REM
REM Usage:
REM   scripts\run_table1_all.bat
REM
REM Optional environment overrides:
REM   set DATA_PATH=data\benchmark\inputs_10.json
REM   set SAVE_EVERY=20
REM   set RESUME=1
REM   set GENERATE_REPORT=1
REM   set USE_MEMORY=1
REM   set AGENT_VERSION=full
REM   set KNOWLEDGE_CUTOFF_DATE=2024-01-01
REM   scripts\run_table1_all.bat

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\.." >nul

if not defined DATA_PATH set "DATA_PATH=data\benchmark\inputs.json"
if not defined SAVE_EVERY set "SAVE_EVERY=10"
if not defined RESUME set "RESUME=0"
if not defined GENERATE_REPORT set "GENERATE_REPORT=0"
if not defined USE_MEMORY set "USE_MEMORY=0"
if not defined AGENT_VERSION set "AGENT_VERSION=eg"
if not defined KNOWLEDGE_CUTOFF_DATE set "KNOWLEDGE_CUTOFF_DATE="
if not defined LOG_DIR set "LOG_DIR=results\table1_logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%DATA_PATH%" (
  echo ERROR: data file not found: %DATA_PATH%
  popd >nul
  exit /b 1
)

set "BACKBONES=gpt-4o llama-4 kimi-k2 gpt5 claude gemini"
set "METHODS=direct cot multiagent"

echo Table 1 experiment batch
echo Root: %CD%
echo Data: %DATA_PATH%
echo Logs: %LOG_DIR%
echo TreatAgent version: %AGENT_VERSION%
echo.

for %%B in (%BACKBONES%) do (
  for %%M in (%METHODS%) do (
    call :run_one %%B %%M
    if errorlevel 1 (
      echo.
      echo ERROR: task failed for backbone=%%B method=%%M
      echo See logs in %LOG_DIR%
      popd >nul
      exit /b 1
    )
  )
)

echo.
echo All Table 1 tasks completed.
echo Detailed result JSON files are under results\^<backbone^>\.
echo Task logs are under %LOG_DIR%\.
popd >nul
exit /b 0

:run_one
set "BACKBONE=%~1"
set "METHOD=%~2"

set "METHOD_LABEL=%METHOD%"
if "%METHOD%"=="direct" set "METHOD_LABEL=One-shot learning"
if "%METHOD%"=="cot" set "METHOD_LABEL=Zero-shot CoT"
if "%METHOD%"=="multiagent" set "METHOD_LABEL=TreatAgent v2"

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%T"
set "LOG_FILE=%LOG_DIR%\%TIMESTAMP%_%BACKBONE%_%METHOD%.log"

set "COMMON_ARGS=--json_path %DATA_PATH% --save_every %SAVE_EVERY%"
if "%RESUME%"=="1" set "COMMON_ARGS=%COMMON_ARGS% --resume"
if not "%KNOWLEDGE_CUTOFF_DATE%"=="" set "COMMON_ARGS=%COMMON_ARGS% --knowledge_cutoff_date %KNOWLEDGE_CUTOFF_DATE%"

set "EXTRA_ARGS="
if "%METHOD%"=="multiagent" (
  set "EXTRA_ARGS=!EXTRA_ARGS! --agent_version %AGENT_VERSION%"
  if "%GENERATE_REPORT%"=="1" set "EXTRA_ARGS=!EXTRA_ARGS! --generate_report"
  if "%USE_MEMORY%"=="1" set "EXTRA_ARGS=!EXTRA_ARGS! --use_memory"
)

echo ============================================================
echo Starting Table 1 task
echo Model:  %BACKBONE%
echo Method: %METHOD_LABEL% ^(%METHOD%^)
if "%METHOD%"=="multiagent" echo Version: %AGENT_VERSION%
echo Data:   %DATA_PATH%
echo Log:    %LOG_FILE%
echo ============================================================

python -m treatagent.cli %COMMON_ARGS% --method %METHOD% --backbone %BACKBONE% %EXTRA_ARGS% > "%LOG_FILE%" 2>&1
set "TASK_STATUS=%ERRORLEVEL%"

type "%LOG_FILE%"

if not "%TASK_STATUS%"=="0" exit /b %TASK_STATUS%
echo Finished: %BACKBONE% / %METHOD%
echo.
exit /b 0
