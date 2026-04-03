"""
LOGIIK launcher — run this once to start everything.

    python start.py

It will:
  1. Install any missing Python packages automatically.
  2. Run an environment check.
  3. Start the LOGIIK API server in the background.
  4. Open the dashboard in your default browser.

Leave this terminal window open while you use LOGIIK.
Press Ctrl+C to stop.
"""
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PORT = 8001
URL  = f"http://localhost:{PORT}"
DASHBOARD = Path(__file__).parent / "logiik" / "dashboard" / "index.html"


def install_dependencies():
    req = Path(__file__).parent / "requirements.txt"
    if req.exists():
        print("Checking dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req), "-q"]
        )


def wait_for_server(timeout=60):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{URL}/logiik/health", timeout=2)
            return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    install_dependencies()

    print("Starting LOGIIK server...")
    root = Path(__file__).parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "logiik.api.endpoints:app",
         "--host", "0.0.0.0",
         "--port", str(PORT)],
        cwd=root,
        env=env,
    )

    print(f"Waiting for server at {URL} ...")
    if wait_for_server():
        print(f"LOGIIK is running at {URL}")
    else:
        print(f"Server may still be starting — opening dashboard anyway...")

    # Open dashboard HTML directly (no extra server needed)
    if DASHBOARD.exists():
        webbrowser.open(DASHBOARD.as_uri())
    else:
        webbrowser.open(URL)

    print("\nLeave this window open. Press Ctrl+C to stop.\n")
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nStopping LOGIIK...")
        server.terminate()
        server.wait()
        print("Done.")


if __name__ == "__main__":
    main()
