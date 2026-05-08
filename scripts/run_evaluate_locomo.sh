#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/evaluate_locomo.py" \
  --dataset "data/locomo10.json" \
  --backend "openai" \
  --model_name "gpt-4.1-mini" \
  --embed_model "BAAI/bge-small-en-v1.5" \
  --api_key "${OPENAI_API_KEY:-}" \
  --embed_api_key "${OPENAI_API_KEY:-}" \
  --base_url "${OPENAI_BASE_URL:-}" \
  --embed_base_url "${OPENAI_BASE_URL:-}" \
  --retrieve_k 15 \
  --retrieve_k_rough 30 \
  --temperature 0.5 \
  --log_name "locomo_lancedb"
