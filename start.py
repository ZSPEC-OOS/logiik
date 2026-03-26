"""
NERO launcher — run this once to start everything.

    python start.py

It will:
  1. Install any missing Python packages automatically.
  2. Start the NERO server in the background.
  3. Open the dashboard in your default browser.

Leave this terminal window open while you use NERO.
Press Ctrl+C to stop.
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PORT = 8000
URL  = f"http://localhost:{PORT}"


def install_dependencies():
    root = Path(__file__).parent

    # Install third-party requirements (no -q so progress is visible)
    req = root / "requirements.txt"
    if req.exists():
        print("Installing dependencies (this can take several minutes the first time)…")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req)],
        )
        if result.returncode != 0:
            print("Warning: some packages may not have installed correctly.")


def wait_for_server(timeout=60):
    import urllib.request
    deadline = time.time() + timeout
    # Try both localhost and 127.0.0.1 in case one resolves differently on Windows
    urls = [f"{URL}/health", f"http://127.0.0.1:{PORT}/health"]
    while time.time() < deadline:
        for u in urls:
            try:
                urllib.request.urlopen(u, timeout=2)
                return True
            except Exception:
                pass
        time.sleep(1)
    return False


def main():
    install_dependencies()

    print("Starting NERO server…")
    root = Path(__file__).parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "cognita.api.server:app",
            "--host", "0.0.0.0",
            "--port", str(PORT),
        ],
        cwd=root,
        env=env,
    )

    print(f"Waiting for server to be ready at {URL} …")
    if wait_for_server():
        print(f"NERO is running. Opening {URL} …")
    else:
        print(f"Opening {URL} — server may still be starting up…")
    webbrowser.open(URL)

    print("\nLeave this window open while you use NERO. Press Ctrl+C to stop.\n")
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nStopping NERO…")
        server.terminate()
        server.wait()
        print("Done.")


if __name__ == "__main__":
    main()
