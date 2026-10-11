#!/usr/bin/env bash
# Pull every image the local deployment needs while the network is good (at home, not on
# conference Wi-Fi). install.sh later loads them from the local Docker cache into Minikube; the
# chart and the orchestrator both use imagePullPolicy IfNotPresent, so nothing is pulled again.
set -euo pipefail
source "$(dirname "$0")/versions.env"

for image in \
  "ghcr.io/jetbrains-research/idegym/orchestrator:${IDEGYM_VERSION}" \
  "ghcr.io/jetbrains-research/idegym/watcher:${IDEGYM_VERSION}" \
  "ghcr.io/jetbrains-research/idegym/server-debian-bookworm-20250520-slim:${IDEGYM_VERSION}" \
  "$POSTGRES_IMAGE"; do
  docker pull "$image"
done
helm pull "$IDEGYM_CHART" --version "$IDEGYM_VERSION" --destination "$(dirname "$0")"
