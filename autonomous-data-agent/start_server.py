#!/usr/bin/env python3
"""
Start the FastAPI web server for the Autonomous Data Cleaning Agent
Access at: http://localhost:8000
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 80)
    print("[*] Starting Autonomous Data Cleaning Agent")
    print("=" * 80)
    print()
    print("📱 WEB INTERFACE: http://localhost:8000")
    print("📖 API DOCS: http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    print("=" * 80)
    
    # Change to project root
    os.chdir(Path(__file__).parent)
    
    # Run uvicorn with increased timeouts for large file processing
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--timeout-keep-alive", "120",
            "--timeout-graceful-shutdown", "120"
        ])
    except KeyboardInterrupt:
        print("\n[*] Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
