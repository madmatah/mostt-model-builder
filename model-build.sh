#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT

exec python3 -I -S -B "${REPO_ROOT}/src/mostt_model_builder/bootstrap.py" mostt_model_builder.host_main "$@"
