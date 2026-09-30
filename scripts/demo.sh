#!/usr/bin/env bash
# One command: generate data -> train -> serve -> open dashboard.
# Phase 0: only the serve step exists. Later phases add the generate/train steps here.
set -euo pipefail
cd "$(dirname "$0")/.."

make setup-py setup-web

echo "[demo] TODO(Phase 1): generate synthetic data"
echo "[demo] TODO(Phase 3): train Module B"

make web-build
echo "[demo] serving API + dashboard on http://127.0.0.1:${API_PORT:-8000}"
if command -v xdg-open >/dev/null 2>&1; then
  (sleep 2 && xdg-open "http://127.0.0.1:${API_PORT:-8000}/" >/dev/null 2>&1) &
fi
exec .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port "${API_PORT:-8000}"
