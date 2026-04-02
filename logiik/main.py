"""
Logiik Entry Point.

Usage:
    python logiik/main.py --mode api
    python logiik/main.py --mode session
    python logiik/main.py --mode check
    python logiik/main.py --mode test
"""
import sys
import argparse


def run_api():
    """Start the Logiik FastAPI server."""
    import uvicorn
    from logiik.config import CONFIG
    from logiik.utils.logging import log_event
    port = CONFIG.get("api", {}).get("port", 8001)
    log_event("main", f"Starting Logiik API on port {port}")
    uvicorn.run(
        "logiik.api.endpoints:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


def run_session():
    """Start the GPU session query server."""
    from logiik.utils.logging import log_event
    log_event("main", "Starting Logiik session manager...")
    import logiik.session_manager.query_server as qs
    from logiik.config import CONFIG
    port = CONFIG.get("session", {}).get("api_port", 5000)
    qs.app.run(host="0.0.0.0", port=port, debug=False)


def run_check():
    """Run environment verification."""
    from logiik.utils.env_check import run_all_checks
    ok = run_all_checks()
    sys.exit(0 if ok else 1)


def run_tests():
    """Run full test suite via pytest."""
    try:
        import pytest
    except ImportError:
        print("pytest not installed. Run: pip install pytest")
        sys.exit(1)
    result = pytest.main([
        "logiik/tests/test_modules.py",
        "-v",
        "--tb=short",
    ])
    sys.exit(result)


def main():
    parser = argparse.ArgumentParser(
        description="Logiik — Scientific Reasoning AI Framework"
    )
    parser.add_argument(
        "--mode",
        choices=["api", "session", "check", "test"],
        default="check",
        help=(
            "api     = start FastAPI server + dashboard\n"
            "session = start GPU session query server\n"
            "check   = run environment verification\n"
            "test    = run full test suite"
        )
    )
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    elif args.mode == "session":
        run_session()
    elif args.mode == "check":
        run_check()
    elif args.mode == "test":
        run_tests()


if __name__ == "__main__":
    main()
