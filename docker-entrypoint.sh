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
    exec streamlit run cognita/dashboard/app.py \
      --server.port 8501 \
      --server.address 0.0.0.0 \
      --server.headless true
    ;;
  *)
    echo "Unknown service: $SERVICE. Use SERVICE=api or SERVICE=dashboard"
    exit 1
    ;;
esac
