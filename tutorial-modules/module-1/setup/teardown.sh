#!/usr/bin/env bash
# Delete the whole cluster. Deleting only the release would keep PostgreSQL's volume, whose
# password no longer matches a reinstalled chart's freshly generated secret.
set -euo pipefail
minikube delete
