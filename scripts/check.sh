#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly STEPS=(format lint types imports tests)
readonly UV_INSTALL="curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/0.13.7/install.sh | sh"

fail() {
    printf 'check: %s\n' "$1" >&2
    exit 1
}

# A missing tool would make a step fail for a reason unrelated to the change,
# so it stops the gate before any step runs.
require_tools() {
    command -v uv >/dev/null || fail "missing uv, which installs with: ${UV_INSTALL}"
}

run_locked() {
    (cd "${REPO_ROOT}" && uv run --locked "$@")
}

# The package is not installed in the environment, since the project builds
# none: import-linter finds it on the path, as mypy and pytest do.
run_step() {
    case "$1" in
    format) run_locked ruff format --check ;;
    lint) run_locked ruff check ;;
    types) run_locked mypy ;;
    imports) PYTHONPATH="${REPO_ROOT}/src" run_locked lint-imports ;;
    tests) run_locked pytest ;;
    esac
}

is_step() {
    local step
    for step in "${STEPS[@]}"; do
        [[ "${step}" == "$1" ]] && return 0
    done
    return 1
}

main() {
    local requested=("$@") step
    ((${#requested[@]} > 0)) || requested=("${STEPS[@]}")
    for step in "${requested[@]}"; do
        is_step "${step}" || fail "unknown step ${step}; the steps are: ${STEPS[*]}"
    done
    require_tools
    for step in "${STEPS[@]}"; do
        [[ " ${requested[*]} " == *" ${step} "* ]] || continue
        printf '== %s\n' "${step}"
        run_step "${step}" || fail "step ${step} failed"
    done
    printf 'check: every step passed: %s\n' "${requested[*]}"
}

main "$@"
