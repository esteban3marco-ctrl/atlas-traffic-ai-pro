@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

cls
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                                                              ║
echo  ║          ATLAS Pro  ─  AI Traffic Intelligence v4.0         ║
echo  ║          Powered by Google Stitch + Gemini 2.5              ║
echo  ║                                                              ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

:: ── Step 1: Check Python ──────────────────────────────────────────────
echo [1/5] Checking Python...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Python not found. Please install Python 3.10+ from https://python.org
    pause & exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo       Found: %%i

:: ── Step 2: Check Node.js ────────────────────────────────────────────
echo.
echo [2/5] Checking Node.js (required for Stitch MCP)...
node --version > nul 2>&1
if %errorlevel% neq 0 (
    echo  WARNING: Node.js not found. Stitch MCP will not be available.
    echo  Install from: https://nodejs.org
    set NODE_OK=0
) else (
    for /f "tokens=*" %%i in ('node --version 2^>^&1') do echo       Found: Node.js %%i
    set NODE_OK=1
)

:: ── Step 3: Virtual environment ──────────────────────────────────────
echo.
echo [3/5] Setting up Python environment...
if not exist ".venv\Scripts\activate.bat" (
    echo       Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo  ERROR: Failed to create virtual environment.
        pause & exit /b 1
    )
    echo       Virtual environment created.
) else (
    echo       Virtual environment found.
)
call .venv\Scripts\activate.bat

:: ── Step 4: Install Python deps ──────────────────────────────────────
echo.
echo [4/5] Installing Python dependencies...
pip install -q --upgrade pip
pip install -q fastapi "uvicorn[standard]" numpy pydantic
if %errorlevel% neq 0 (
    echo  ERROR: Failed to install dependencies.
    pause & exit /b 1
)
echo       All dependencies installed.

:: ── Step 5: Stitch MCP check ─────────────────────────────────────────
echo.
echo [5/5] Checking Stitch MCP...
if "!NODE_OK!"=="1" (
    where npx > nul 2>&1
    if !errorlevel! equ 0 (
        echo       Stitch MCP available via npx @_davideast/stitch-mcp
        echo       NOTE: Run 'npx @_davideast/stitch-mcp init' once to authenticate.
        set STITCH_OK=1
    ) else (
        echo       npx not available.
        set STITCH_OK=0
    )
) else (
    set STITCH_OK=0
)

:: ── Launch ───────────────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                                                              ║
echo  ║  ✓ ATLAS Pro Dashboard is starting...                       ║
echo  ║                                                              ║
echo  ║  Dashboard  →  http://localhost:8000                        ║
echo  ║  API Docs   →  http://localhost:8000/docs                   ║
echo  ║  REST API   →  http://localhost:8000/api/status             ║
echo  ║                                                              ║
echo  ║  Press Ctrl+C to stop                                       ║
echo  ║                                                              ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

:: Open browser after 2 seconds
ping -n 3 127.0.0.1 > nul
start "" "http://localhost:8000"

:: Run server
python dashboard_server.py

echo.
echo  Server stopped.
pause
