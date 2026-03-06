#!/usr/bin/env bash
set -euo pipefail

HOST="${TS_HOST:-0.0.0.0}"
PORT="${TS_PORT:-8000}"
WITH_BASIC="${WITH_BASIC:-1}"
SERVICE_MODULE="${SERVICE_MODULE:-service_tu4:app}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "[ERROR] uvicorn not found. Please install fastapi/uvicorn first." >&2
  exit 1
fi

TRADE_DATE="$(TZ=Asia/Shanghai date +%Y%m%d)"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
REFRESH_URL="http://127.0.0.1:${PORT}/refresh_universe?trade_date=${TRADE_DATE}&with_basic=${WITH_BASIC}"

if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
  echo "[INFO] service already running on port ${PORT}."
else
  echo "[INFO] starting uvicorn ${SERVICE_MODULE} on ${HOST}:${PORT} ..."
  nohup uvicorn "$SERVICE_MODULE" --host "$HOST" --port "$PORT" > uvicorn.log 2>&1 &
  SERVER_PID=$!
  echo "[INFO] uvicorn pid=${SERVER_PID}, waiting for health..."

  for _ in {1..40}; do
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done

  if ! curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "[ERROR] service not ready. check ${SCRIPT_DIR}/uvicorn.log" >&2
    exit 1
  fi
fi

echo "[INFO] refreshing universe for trade_date=${TRADE_DATE}, with_basic=${WITH_BASIC} ..."
curl -fsS -X POST "$REFRESH_URL"
echo

echo "[DONE] service ready: http://127.0.0.1:${PORT}"
echo "[DONE] dashboard:    http://127.0.0.1:${PORT}/"
