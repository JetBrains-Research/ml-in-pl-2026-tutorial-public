#!/usr/bin/env bash
# Deploy IdeGYM into a fresh Minikube cluster using the published chart and prebuilt images.
# Follows the IdeGYM local deployment guide (website/docs/reference/local_deployment.md).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/versions.env"

# Sandboxes run with the default container runtime: no gVisor, and no ingress, since the
# orchestrator is reached through `kubectl port-forward`.
minikube start \
  --cpus="$MINIKUBE_CPUS" \
  --memory="$MINIKUBE_MEMORY" \
  --kubernetes-version="$KUBERNETES_VERSION"

# Images pulled by prepull.sh are copied from the host instead of downloaded again.
for ref in \
  "ghcr.io/jetbrains-research/idegym/orchestrator:${IDEGYM_VERSION}" \
  "ghcr.io/jetbrains-research/idegym/watcher:${IDEGYM_VERSION}" \
  "ghcr.io/jetbrains-research/idegym/server-debian-bookworm-20250520-slim:${IDEGYM_VERSION}" \
  "$POSTGRES_IMAGE"; do
  if docker image inspect "$ref" >/dev/null 2>&1; then
    minikube image load "$ref"
  fi
done

kubectl create namespace idegym --dry-run=client -o yaml | kubectl apply -f -

chart="$here/idegym-${IDEGYM_VERSION}.tgz"
[ -f "$chart" ] || chart="$IDEGYM_CHART"
helm upgrade --install idegym "$chart" --version "$IDEGYM_VERSION" -n idegym --wait --timeout 10m

echo
echo "IdeGYM is running. In a separate terminal, keep this open:"
echo "  kubectl port-forward svc/idegym 8000:80 -n idegym"
echo "Then check: curl http://localhost:8000/health"
