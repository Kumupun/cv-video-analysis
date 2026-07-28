#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH=${1:?Usage: backend/scripts/smoke_test.sh /path/to/video.mp4}
API_URL=${API_URL:-http://localhost:8000/api/v1}

response=$(curl -fsS -X POST "$API_URL/process" -F "file=@${VIDEO_PATH}")
echo "$response"
task_id=$(python -c 'import json,sys; print(json.load(sys.stdin)["task_id"])' <<<"$response")

while true; do
  status=$(curl -fsS "$API_URL/status/$task_id")
  echo "$status"
  stage=$(python -c 'import json,sys; print(json.load(sys.stdin)["stage"])' <<<"$status")
  case "$stage" in
    completed)
      curl -fsS "$API_URL/results/$task_id"
      echo
      break
      ;;
    failed)
      exit 1
      ;;
  esac
  sleep 2
done
