#!/usr/bin/env bash
# The whole gate runs in well under a minute, so every commit runs all of it
# rather than the steps its files would select.
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${REPO_ROOT}/scripts/check.sh"
