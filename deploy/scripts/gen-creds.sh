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
# All Engram services (mirrors launcher/registry.py, plus meta-programmer
# which launcher/registry.py intentionally omits — it needs the Docker
# socket and only runs via `docker compose --profile extra up` — but which
# still needs a NATS identity for that Docker deployment).
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
    meta-programmer
)

# ---------------------------------------------------------------------------
# Subject allowlists per service — least privilege, matching *actual*
# publish()/subscribe()/request() call sites as of issue #184's audit (not
# the ADR 0001 matrix's target/aspirational state, where the two differ —
# several ADR-documented subjects are not yet wired up in code, and a few
# real call sites aren't in the ADR at all; re-audit by grepping each
# service's src/ for `.publish(`/`.subscribe(`/`.request(` if this drifts).
# Per-trace/per-channel dynamic subjects (e.g. `outcome.{trace_id}`) are
# granted via their wildcard prefix, matching the existing decision.>/
# code.decision.> convention. Request-reply replies (`is_request_handler=
# True` + `msg.respond()`) need no dedicated subject beyond `_INBOX.>` —
# the reply goes to the caller's ephemeral inbox, not a fixed subject.
# Format: pub:<subject1>,<subject2>  sub:<subject1>,<subject2>
# Wildcards: > (match rest) and * (one token).
# ---------------------------------------------------------------------------
declare -A SVC_PUB SVC_SUB

# kernel — sole authority for decision.>/code.decision.>/policy.restrict/
# cognitive.response.validated (ADR 0001 §3); everything else here is a
# broadcast notification or a request (beliefs.query.request) it issues.
SVC_PUB[kernel]="decision.>,code.decision.>,policy.restrict,policy.restrict.status,policy.rollback.status,policy.update.status,policy.profile.status,cognitive.response.validated,cognitive.response.rejected,cognitive.execute,speech.execute,planner.mode,safety.halt.status,safety.deny_escalation,kernel.heartbeat,kernel.decision_rates,beliefs.query.request,_INBOX.>"
SVC_SUB[kernel]="proposal.new,code.proposal,kernel.status,policy.load_profile,policy.restrict.request,policy.rollback,policy.update,cognitive.response.validate,safety.halt,safety.resume,dlq.>,_INBOX.>"

# safety-supervisor — request-reply risk analysis; safety.analysis.*.<trace>
# is a real fallback publish path (safety_supervisor/service.py) used when
# the request has no reply-to (backward compat), not just the msg.respond().
SVC_PUB[safety-supervisor]="safety.analysis.action.>,safety.analysis.code.>,_INBOX.>"
SVC_SUB[safety-supervisor]="safety.analyze.action,safety.analyze.code,safety.status,_INBOX.>"

SVC_PUB[beliefs]="beliefs.query.response,_INBOX.>"
SVC_SUB[beliefs]="beliefs.add_node,beliefs.add_edge,beliefs.update,beliefs.query,beliefs.contradictions,beliefs.query.request,_INBOX.>"

# overrides — human override flow; publishes proposal.new like any proposal
# source and waits on decision.<trace> (EventBus.wait_for_decision).
SVC_PUB[overrides]="proposal.new,override.result.>,override.applied.>,planner.mode,cache.setting,autopilot.setting,planner.priority_threshold,_INBOX.>"
SVC_SUB[overrides]="override.request,override.status,decision.>,_INBOX.>"

SVC_PUB[planner]="proposal.new,outcome.>,approval.request,_INBOX.>"
SVC_SUB[planner]="observation.*,decision.*,planner.mode,planner.status,_INBOX.>"

# external-api — escalates knowledge conflicts to human approval and asks
# the Kernel to approve outbound queries (EventBus.wait_for_decision).
SVC_PUB[external-api]="proposal.new,approval.request,_INBOX.>"
SVC_SUB[external-api]="external.query,external.status,decision.>,_INBOX.>"

# neuromorphic (the brain) — largest surface: sensory intake, motor
# feedback loop (incl. MuJoCo body + real-actuator adapters, which run
# in-process under neuromorphic/src/neuromorphic/actuators/), governance
# proposals, and the memory/beliefs/knowledge-gap side channels.
SVC_PUB[neuromorphic]="proposal.new,memory.store,neuromorphic.metrics,neuromorphic.concept.result,gateway.restart.request,beliefs.add_node,approval.request,knowledge.gap,safety.watchdog.status,policy.restrict.request,mujoco.body.state,motor.execute.>,motor.outcome.>,neuromod.teach.da,body.posture,observation.visual.body,observation.proprioceptive,observation.pain,teach.progress,actuator.heartbeat.>,outcome.>,_INBOX.>"
SVC_SUB[neuromorphic]="observation.>,neuromorphic.status,neuromorphic.drives.event,neuromorphic.teach,neuromorphic.train_bulk,neuromorphic.concept.probe,cognitive.response.validated,motor.outcome.>,neuromod.teach.da,body.posture,decision.>,approval.response.>,motor.guidance,actuator.heartbeat.>,training.session.complete,_INBOX.>"

# dashboard — publish surface is the operator's hands (sensory-gateway
# commands, motor guidance, safety halt/resume); subscribe stays the one
# intentional broad `>` grant (read-everything for the live UI, ADR 0001
# §4) since it conveys no publish rights.
SVC_PUB[dashboard]="sensory.gateway.command,motor.guidance,neuromorphic.concept.probe,approval.response.>,safety.halt,safety.resume,observation.text,_INBOX.>"
SVC_SUB[dashboard]=">"

SVC_PUB[memory]="memory.recall.result.>,_INBOX.>"
SVC_SUB[memory]="memory.store,memory.query,memory.recall,_INBOX.>"

SVC_PUB[cache]="llm.cache.>,_INBOX.>"
SVC_SUB[cache]="cache.query,cache.setting,cache.status,autopilot.setting,code.deployed,override.applied.*,task.saved,_INBOX.>"

SVC_PUB[coordinator]="proposal.new,task.result,demo.started,demo.finished,demo.failed,knowledge.gap,learning.started,_INBOX.>"
SVC_SUB[coordinator]="task.request,demo.start,demo.observation,demo.finish,coordinator.status,device.unknown,decision.>,_INBOX.>"

# cognitive-bridge — brain<->Ollama bridge; runs as its own process
# (module neuromorphic.cognitive_bridge) separate from the neuromorphic
# brain, so it gets its own narrower identity.
SVC_PUB[cognitive-bridge]="cognitive.response.validate,_INBOX.>"
SVC_SUB[cognitive-bridge]="cognitive.execute,cognitive.query,_INBOX.>"

SVC_PUB[sensory-gateway]="observation.>,sensory.gateway.status,video.training.status,training.session.complete,device.unknown,_INBOX.>"
SVC_SUB[sensory-gateway]="sensory.gateway.command,neuromorphic.metrics,device.driver.ready,_INBOX.>"

# meta-programmer — previously had NO entry here at all (a real gap, not
# just a broad one); waits on code.decision.> (EventBus.wait_for_decision
# with code=True) for its own code proposals.
SVC_PUB[meta-programmer]="code.proposal,approval.request,device.driver.ready,deploy.rolled_back,knowledge.gap.result.>,_INBOX.>"
SVC_SUB[meta-programmer]="knowledge.gap,metaprogrammer.status,approval.response.*,code.decision.>,_INBOX.>"

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
