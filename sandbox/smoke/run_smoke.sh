#!/usr/bin/env bash
#
# Smoke test for activelearning-sandbox:latest.
#
# Proves the sandbox image can run untrusted generated tests under the SAME
# hardening the SandboxManager applies at runtime (no network, read-only root,
# all caps dropped, no-new-privileges, tmpfs /tmp, pid/mem/cpu limits) and that
# it correctly reports BOTH pass and fail.
#
# Usage:
#   sandbox/smoke/run_smoke.sh            # build image if missing, then test
#   SKIP_BUILD=1 sandbox/smoke/run_smoke.sh
#
# Exits non-zero (failing CI) if either case behaves unexpectedly.

set -euo pipefail

IMAGE="activelearning-sandbox:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# These flags mirror meta-programmer/src/meta_programmer/sandbox_manager.py.
HARDENING=(
  --rm
  --network none
  --read-only
  --cap-drop ALL
  --security-opt no-new-privileges
  --pids-limit 100
  --memory 512m
  --memory-swap 512m
  --cpus 0.5
  --tmpfs /tmp:size=50M
)

build_image() {
  if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "SKIP_BUILD=1 — assuming ${IMAGE} already exists"
    return
  fi
  echo "==> Building ${IMAGE} via docker compose build-only profile"
  ( cd "${REPO_ROOT}" && docker compose --profile build-only build sandbox-base )
}

# Run pytest on a fixture file inside a hardened container; echo the exit code.
run_case() {
  local test_file="$1"
  local rc=0
  docker run "${HARDENING[@]}" \
    -v "${SCRIPT_DIR}:/sandbox:ro" \
    "${IMAGE}" \
    pytest "/sandbox/${test_file}" --timeout=5 --tb=short -v || rc=$?
  echo "${rc}"
}

main() {
  build_image

  echo "==> Case 1: passing test set must exit 0"
  local pass_rc
  pass_rc="$(run_case test_pass.py | tail -n1)"
  if [[ "${pass_rc}" != "0" ]]; then
    echo "FAIL: expected passing set to exit 0, got ${pass_rc}" >&2
    exit 1
  fi
  echo "    OK (exit 0)"

  echo "==> Case 2: failing test set must exit non-zero"
  local fail_rc
  fail_rc="$(run_case test_fail.py | tail -n1)"
  if [[ "${fail_rc}" == "0" ]]; then
    echo "FAIL: expected failing set to exit non-zero, got 0" >&2
    exit 1
  fi
  echo "    OK (exit ${fail_rc})"

  echo "==> Case 3: network must be unreachable inside the sandbox"
  local net_rc=0
  docker run "${HARDENING[@]}" "${IMAGE}" \
    python -c "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)" \
    >/dev/null 2>&1 || net_rc=$?
  if [[ "${net_rc}" == "0" ]]; then
    echo "FAIL: network reachable inside sandbox — isolation broken" >&2
    exit 1
  fi
  echo "    OK (network blocked, exit ${net_rc})"

  echo "==> Case 4: root filesystem must be read-only"
  local ro_rc=0
  docker run "${HARDENING[@]}" "${IMAGE}" \
    sh -c "echo nope > /should_fail.txt" >/dev/null 2>&1 || ro_rc=$?
  if [[ "${ro_rc}" == "0" ]]; then
    echo "FAIL: wrote to read-only root filesystem — isolation broken" >&2
    exit 1
  fi
  echo "    OK (read-only enforced, exit ${ro_rc})"

  echo "All sandbox smoke checks passed."
}

main "$@"
