#!/usr/bin/env python3
"""SciAgent One-Command Local Startup Script.

Checks Ollama status, starts FastAPI backend on port 8080,
starts Vite frontend on port 8000, and opens http://localhost:8000 in your browser.
"""

import os
import sys
import time
import subprocess
import webbrowser
import urllib.request

BACKEND_PORT = 8080
FRONTEND_PORT = 8000
OLLAMA_URL = "http://localhost:11434/api/tags"

def check_ollama() -> bool:
    print("[1/4] Checking Ollama local LLM daemon...")
    try:
        with urllib.request.urlopen(OLLAMA_URL, timeout=3) as resp:
            if resp.status == 200:
                print("  ✓ Ollama daemon is running on http://localhost:11434")
                return True
    except Exception:
        pass
    print("  ⚠ Ollama is not active or unreachable. SciAgent will operate in deterministic offline rule-based mode.")
    return False

def check_port(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}", timeout=2) as resp:
            return True
    except Exception:
        return False

def main():
    print("==================================================================")
    print("           SCIENTIFIC AGENT (SciAgent) LOCAL LAUNCHER             ")
    print("==================================================================")

    # 1. Ollama status
    check_ollama()

    # 2. Start FastAPI Backend if not already running
    backend_proc = None
    if not check_port(BACKEND_PORT):
        print(f"[2/4] Starting FastAPI backend on http://localhost:{BACKEND_PORT}...")
        backend_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        # Wait for backend to be ready
        for _ in range(25):
            time.sleep(1)
            if check_port(BACKEND_PORT):
                print(f"  ✓ FastAPI backend is live at http://localhost:{BACKEND_PORT}")
                break
        else:
            print("  ⚠ Backend took longer than expected to report status, continuing...")
    else:
        print(f"[2/4] ✓ FastAPI backend is already running on http://localhost:{BACKEND_PORT}")

    # 3. Start Frontend if not already running
    frontend_proc = None
    if not check_port(FRONTEND_PORT):
        print(f"[3/4] Starting Vite React frontend on http://localhost:{FRONTEND_PORT}...")
        shell_cmd = "npm run dev -- --port 8000 --host" if sys.platform != "win32" else "npm.cmd run dev -- --port 8000 --host"
        frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
        frontend_proc = subprocess.Popen(
            shell_cmd,
            cwd=frontend_dir,
            shell=True,
        )
        for _ in range(15):
            time.sleep(1)
            if check_port(FRONTEND_PORT):
                print(f"  ✓ Frontend is live at http://localhost:{FRONTEND_PORT}")
                break
    else:
        print(f"[3/4] ✓ Frontend is already running on http://localhost:{FRONTEND_PORT}")

    # 4. Open browser
    print(f"[4/4] Opening browser to http://localhost:{FRONTEND_PORT}...")
    webbrowser.open(f"http://localhost:{FRONTEND_PORT}")

    print("\n SciAgent is fully operational!")
    print(f"  - Web Interface: http://localhost:{FRONTEND_PORT}")
    print(f"  - API Docs:      http://localhost:{BACKEND_PORT}/docs")
    print("Press Ctrl+C to terminate all services.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down SciAgent services...")
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        print("Goodbye!")

if __name__ == "__main__":
    main()
