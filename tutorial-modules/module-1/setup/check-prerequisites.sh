#!/usr/bin/env bash
# Run before the workshop: fails fast on anything missing, so setup time is spent on IdeGYM.
set -euo pipefail

status=0
for tool in docker minikube kubectl helm uv; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "ok       $tool"
  else
    echo "MISSING  $tool"
    status=1
  fi
done
if ! docker info >/dev/null 2>&1; then
  echo "MISSING  a running Docker daemon (start Docker Desktop or dockerd)"
  status=1
fi
exit "$status"
