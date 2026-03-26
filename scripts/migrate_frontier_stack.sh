#!/usr/bin/env bash
set -euo pipefail

# NERO migration helper: legacy -> frontier-aligned open-weight MoE stack.
# Usage:
#   bash scripts/migrate_frontier_stack.sh [--with-ollama-pull]

WITH_PULL="false"
if [[ "${1:-}" == "--with-ollama-pull" ]]; then
  WITH_PULL="true"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${ROOT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"

echo "[1/6] Backing up repository to: ${BACKUP_DIR}"
cp -r "${ROOT_DIR}" "${BACKUP_DIR}"

echo "[2/6] Installing frontier orchestration/runtime dependencies"
python -m pip install --upgrade \
  vllm \
  langgraph \
  langgraph-checkpoint-sqlite \
  langchain \
  langchain-community \
  langchain-ollama \
  llama-index \
  llama-index-embeddings-huggingface \
  llama-index-vector-stores-chroma \
  pydantic==2.10.0

echo "[3/6] Installing PyTorch CUDA 12.4 wheels"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "[4/6] Ensuring Ollama is installed"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "Ollama already installed."
fi

echo "[5/6] Optionally pulling frontier model weights"
if [[ "${WITH_PULL}" == "true" ]]; then
  ollama pull llama4:maverick || true
  ollama pull deepseek-v3.2 || true
else
  echo "Skipping pull. Re-run with --with-ollama-pull to fetch weights."
fi

echo "[6/6] Migration bootstrap complete"
echo "Next steps:"
echo "  - Review configs/frontier_stack.yaml"
echo "  - Start Ollama: ollama serve"
echo "  - Run graph example: python -m cognita.orchestration.bio_coding_graph"
