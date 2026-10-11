"""Published generic runtimes; task files are supplied by the experiment runners."""

REGISTRY = "ghcr.io/jetbrains-research/ml-in-pl-2026-tutorial-public"
BOOKWORM_IMAGE = f"{REGISTRY}/idegym-server-20250520-slim:1.0.0"
PYCHARM_IMAGE = (
    "europe-west4-docker.pkg.dev/gke-dev-dws-jbr/idegym/ml-in-pl-workshop-pycharm-mcp:5fa379b2f8d4e74aa9fba89eac673fd5"
)

OPENENV_IMAGE = f"{REGISTRY}/openenv-server-tbench2-env:latest"
