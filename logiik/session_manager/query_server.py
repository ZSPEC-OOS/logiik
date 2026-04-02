"""
Logiik Query Server.

Lightweight Flask API wrapping LogiikSession.
Users send queries without touching developer code.

Endpoints:
  GET  /status   — session health and GPU stats
  POST /ask      — submit a query, receive answer
  POST /shutdown — safely terminate session

Run:
    python -m logiik.session_manager.query_server

Then query:
    curl -X POST http://localhost:5000/ask \
         -H 'Content-Type: application/json' \
         -d '{"question": "How does pH affect enzyme folding?"}'
"""
import os
import sys
import signal
from typing import Optional

from flask import Flask, request, jsonify

from logiik.config import CONFIG
from logiik.session_manager.session_manager import LogiikSession
from logiik.session_manager.utils.helpers import SessionLogger

_logger = SessionLogger("query_server")

app = Flask(__name__)

# ─── Session singleton ────────────────────────────────────────────────────────
# Initialised at startup — model loads on first /ask call.

_session: Optional[LogiikSession] = None


def get_session() -> LogiikSession:
    global _session
    if _session is None:
        _session = LogiikSession()
    return _session


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    """
    Session health check.
    Returns model load state, GPU stats, query count, idle time.
    """
    try:
        return jsonify(get_session().get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    Submit a query to Logiik.

    Request body (JSON):
        {"question": "your question here"}

    Response:
        {
          "answer": "...",
          "context_chunks": [...],
          "latency_ms": 123.4,
          "query_count": 1
        }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    question = data.get("question", "").strip()
    if not question:
        return jsonify(
            {"error": "Field 'question' is required and cannot be empty."}
        ), 400

    try:
        result = get_session().query(question)
        return jsonify(result)
    except RuntimeError as e:
        # Session shutting down
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        _logger.error(f"/ask error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """
    Safely terminate the GPU session.

    Unloads model, clears VRAM, stops auto-expiry watchdog.
    After calling this endpoint, stop the GPU instance on
    your provider dashboard to avoid unnecessary charges.
    """
    try:
        get_session().shutdown()
        _logger.info("Shutdown via /shutdown endpoint.")

        # Graceful server shutdown after response sent
        def stop_server():
            import time
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

        import threading
        threading.Thread(target=stop_server, daemon=True).start()

        return jsonify({
            "status": "Session terminated. GPU memory freed. "
                      "You can now stop the GPU instance."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = CONFIG.get("session", {}).get("api_port", 5000)
    _logger.info(f"Logiik Query Server starting on port {port}")
    _logger.info("Endpoints: GET /status  POST /ask  POST /shutdown")
    _logger.info(
        "Model loads on first /ask call. "
        "POST /shutdown when done, then stop GPU instance."
    )
    app.run(host="0.0.0.0", port=port, debug=False)
