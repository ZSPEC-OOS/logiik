#!/bin/bash
set -e

SERVICE="${SERVICE:-api}"

case "$SERVICE" in
  api)
    echo "Starting Cognita API server..."
    exec uvicorn cognita.api.server:app --host 0.0.0.0 --port 8000
    ;;
  dashboard)
    echo "Starting Cognita Dashboard..."
    if [ -d "dashboard" ]; then
      exec python -m http.server 8501 --bind 0.0.0.0 --directory dashboard
    else
      echo "Dashboard assets not found at /app/dashboard"
      exit 1
    fi
    ;;
  *)
    echo "Unknown service: $SERVICE. Use SERVICE=api or SERVICE=dashboard"
    exit 1
    ;;
esac
