#!/usr/bin/env bash
# deploy/scripts/gen-creds.sh
#
# Generate per-service NATS NKEY/JWT credentials for Engram.
#
# Prerequisites
# -------------
#   nsc  >= 0.0.35  https://nats-io.github.io/nsc/
#   nats-server     (already managed by run.py / docker)
#
# Install nsc:
#   curl -sfL https://raw.githubusercontent.com/nats-io/nsc/main/install.py | python3
#
# What this script does
# ---------------------
#   1. Creates (or reuses) an operator "ENGRAM" and an account "ENGRAM" under
#      NSC_HOME=.localrun/nsc  (gitignored, stays local).
#   2. For each Engram service, creates a NATS user with subject-level
#      publish/subscribe allow-lists (per ADR 0001-nats-authz.md).
#   3. Exports a .creds file to secrets/<service>.creds  (gitignored).
#   4. Writes .localrun/nats/resolver.conf — a NATS memory-resolver config
#      that run.py uses when starting the local NATS server so that the broker
#      enforces the same identities as production.
#
# Usage
# -----
#   bash deploy/scripts/gen-creds.sh           # generate / refresh all creds
#   bash deploy/scripts/gen-creds.sh kernel    # regenerate one service
#
# After running this script:
#   - python run.py          picks up secrets/<svc>.creds automatically
#   - docker compose up      mounts secrets/ into containers (see docker-compose.yml)
#   - deploy/docker-compose.1m.yml  uses the same secrets/ directory on the server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SECRETS_DIR="$ROOT/secrets"
NSC_HOME="$ROOT/.localrun/nsc"
NATS_DIR="$ROOT/.localrun/nats"
RESOLVER_CONF="$NATS_DIR/resolver.conf"

OPERATOR_NAME="ENGRAM"
ACCOUNT_NAME="ENGRAM"

# ---------------------------------------------------------------------------
# All Engram services (mirrors launcher/registry.py).
# Keys with a dash are normalised to underscores for nsc user names.
# ---------------------------------------------------------------------------
ALL_SERVICES=(
    kernel
    safety-supervisor
    beliefs
    planner
    external-api
    neuromorphic
    dashboard
    memory
    cache
    coordinator
    cognitive-bridge
    sensory-gateway
    overrides
)

# ---------------------------------------------------------------------------
# Subject allowlists per service (ADR 0001 §Subject × service matrix).
# Format: pub:<subject1>,<subject2>  sub:<subject1>,<subject2>
# Wildcards: > (match rest) and * (one token).
# ---------------------------------------------------------------------------
declare -A SVC_PUB SVC_SUB

SVC_PUB[kernel]="decision.>,code.decision.>,policy.*,cognitive.response.validated"
SVC_SUB[kernel]="proposal.new,code.proposal,policy.restrict.request,safety.halt.request,_INBOX.>"

SVC_PUB[safety-supervisor]="safety.halt,risk.score"
SVC_SUB[safety-supervisor]="proposal.new,code.proposal,observation.>,sensor.>,safety.halt.request,_INBOX.>"

SVC_PUB[beliefs]="beliefs.>,cognitive.context"
SVC_SUB[beliefs]="beliefs.query,beliefs.update,learning.signal,_INBOX.>"

SVC_PUB[planner]="proposal.new,learning.signal,observation.>"
SVC_SUB[planner]="decision.>,observation.>,sensor.>,neuromorphic.state,_INBOX.>"

SVC_PUB[external-api]="cognitive.response,cognitive.response.validated"
SVC_SUB[external-api]="cognitive.request,_INBOX.>"

SVC_PUB[neuromorphic]="observation.>,neuromorphic.state,learning.signal,policy.restrict.request,proposal.new"
SVC_SUB[neuromorphic]="decision.>,observation.>,sensor.>,policy.*,safety.halt,cognitive.response.validated,learning.signal,_INBOX.>"

SVC_PUB[dashboard]="operator.control.*"
SVC_SUB[dashboard]="observation.>,neuromorphic.state,decision.>,safety.*,beliefs.>,learning.signal,>"

SVC_PUB[memory]="memory.>"
SVC_SUB[memory]="memory.query,memory.store,observation.>,_INBOX.>"

SVC_PUB[cache]="cache.>"
SVC_SUB[cache]="cache.query,cache.store,_INBOX.>"

SVC_PUB[coordinator]="proposal.new,observation.*"
SVC_SUB[coordinator]="neuromorphic.state,decision.>,observation.>,_INBOX.>"

SVC_PUB[cognitive-bridge]="cognitive.request,learning.signal"
SVC_SUB[cognitive-bridge]="neuromorphic.state,cognitive.response,_INBOX.>"

SVC_PUB[sensory-gateway]="sensor.>,observation.>"
SVC_SUB[sensory-gateway]="_INBOX.>"

SVC_PUB[overrides]="proposal.new,operator.control.*"
SVC_SUB[overrides]="decision.>,policy.*,_INBOX.>"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_nsc_name() {
    # nsc user names cannot contain dashes; map service-name → service_name
    echo "${1//-/_}"
}

require_nsc() {
    if ! command -v nsc &>/dev/null; then
        echo "ERROR: 'nsc' not found. Install it with:"
        echo "  curl -sfL https://raw.githubusercontent.com/nats-io/nsc/main/install.py | python3"
        echo "  # or: brew install nats-io/nats-tools/nsc"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Setup operator + account (idempotent)
# ---------------------------------------------------------------------------
setup_operator_and_account() {
    export NSC_HOME
    mkdir -p "$NSC_HOME"

    if ! nsc describe operator "$OPERATOR_NAME" &>/dev/null; then
        echo "Creating operator: $OPERATOR_NAME"
        nsc add operator --name "$OPERATOR_NAME" --sys
    else
        echo "Operator '$OPERATOR_NAME' already exists — reusing"
    fi

    nsc env --operator "$OPERATOR_NAME" >/dev/null

    if ! nsc describe account "$ACCOUNT_NAME" &>/dev/null; then
        echo "Creating account: $ACCOUNT_NAME"
        nsc add account --name "$ACCOUNT_NAME"
    else
        echo "Account '$ACCOUNT_NAME' already exists — reusing"
    fi
}

# ---------------------------------------------------------------------------
# Generate credentials for one service
# ---------------------------------------------------------------------------
gen_service_creds() {
    local svc="$1"
    local user_name
    user_name="$(_nsc_name "$svc")"
    local creds_file="$SECRETS_DIR/${svc}.creds"

    export NSC_HOME

    # Build allow-pub and allow-sub flags
    local pub_flag="" sub_flag=""
    if [[ -n "${SVC_PUB[$svc]:-}" ]]; then
        # nsc expects comma-separated values in a single flag
        pub_flag="--allow-pub ${SVC_PUB[$svc]}"
    fi
    if [[ -n "${SVC_SUB[$svc]:-}" ]]; then
        sub_flag="--allow-sub ${SVC_SUB[$svc]}"
    fi

    if nsc describe user --account "$ACCOUNT_NAME" --name "$user_name" &>/dev/null; then
        echo "Refreshing creds for $svc (user=$user_name)"
    else
        echo "Creating user for $svc (user=$user_name)"
        # shellcheck disable=SC2086
        nsc add user \
            --account "$ACCOUNT_NAME" \
            --name "$user_name" \
            $pub_flag \
            $sub_flag
    fi

    nsc generate creds \
        --account "$ACCOUNT_NAME" \
        --name "$user_name" \
        > "$creds_file"
    chmod 600 "$creds_file"
    echo "  → $creds_file"
}

# ---------------------------------------------------------------------------
# Write resolver.conf for run.py's local NATS server
# ---------------------------------------------------------------------------
write_resolver_conf() {
    export NSC_HOME
    mkdir -p "$NATS_DIR"

    # Pull the operator JWT and account JWT for the memory resolver
    local operator_jwt account_jwt account_pubkey
    operator_jwt="$(nsc describe operator "$OPERATOR_NAME" --raw 2>/dev/null | tr -d '[:space:]')"
    account_pubkey="$(nsc describe account "$ACCOUNT_NAME" --field sub 2>/dev/null | tr -d '"')"
    account_jwt="$(nsc describe account "$ACCOUNT_NAME" --raw 2>/dev/null | tr -d '[:space:]')"

    cat > "$RESOLVER_CONF" <<EOF
# Auto-generated by deploy/scripts/gen-creds.sh — do not edit by hand.
# Loaded by launcher/nats_server.py when this file exists.

listen: 0.0.0.0:4222
http:   0.0.0.0:8222

jetstream {
  store_dir: "$NATS_DIR/data"
}

operator: $operator_jwt

# Memory resolver — all account JWTs embedded; no external lookup needed.
resolver: MEMORY
resolver_preloads: {
  $account_pubkey: $account_jwt
}
EOF
    echo "Wrote resolver config: $RESOLVER_CONF"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
require_nsc
mkdir -p "$SECRETS_DIR"

setup_operator_and_account

# If a specific service was requested, only generate that one
if [[ "${1:-}" != "" ]]; then
    if [[ -z "${SVC_PUB[$1]:-}" && -z "${SVC_SUB[$1]:-}" ]]; then
        echo "ERROR: Unknown service '$1'. Valid services:"
        printf '  %s\n' "${ALL_SERVICES[@]}"
        exit 1
    fi
    gen_service_creds "$1"
else
    for svc in "${ALL_SERVICES[@]}"; do
        gen_service_creds "$svc"
    done
fi

write_resolver_conf

echo ""
echo "Done. Per-service credentials written to secrets/."
echo ""
echo "Next steps:"
echo "  python run.py                    # picks up secrets/*.creds automatically"
echo "  docker compose up                # mounts secrets/ into containers"
echo ""
echo "To verify identities after startup, check logs for:"
echo "  'Connecting to NATS ... with credentials secrets/<svc>.creds'"
