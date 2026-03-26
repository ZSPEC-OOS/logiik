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

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PORT = 8000
URL  = f"http://localhost:{PORT}"


def install_dependencies():
    req = Path(__file__).parent / "requirements.txt"
    if not req.exists():
        return
    print("Checking dependencies…")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req), "-q"],
        capture_output=True,
    )
    if result.returncode != 0:
        print("Warning: some packages may not have installed correctly.")
        print(result.stderr.decode())


def wait_for_server(timeout=30):
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{URL}/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    install_dependencies()

    print("Starting NERO server…")
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "cognita.api.server:app",
            "--host", "0.0.0.0",
            "--port", str(PORT),
        ],
        cwd=Path(__file__).parent,
    )

    print(f"Waiting for server to be ready at {URL} …")
    if wait_for_server():
        print(f"NERO is running. Opening {URL} …")
        webbrowser.open(URL)
    else:
        print(f"Server did not respond in time. Open {URL} manually.")

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
