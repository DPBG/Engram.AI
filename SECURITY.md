# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it privately by emailing **rio@engram.ai**. Do not open a public issue.

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Security Architecture

- **Kernel**: Immutable safety gate. All motor commands and code proposals pass through the Kernel before execution. Cannot be modified or bypassed.
- **Safety Supervisor**: Risk analysis layer. Evaluates proposals for dangerous patterns.
- **Sandbox**: Ephemeral containers with no network, read-only filesystem, and resource limits for testing generated code.
- **NATS Authentication**: Production deployments use token-based auth today; [ADR 0001](docs/adr/0001-nats-authz.md) defines per-service least-privilege credentials (in progress). Tokens are stored in `.env` (git-ignored) and GitHub Actions secrets.
- **Forged ALLOW regression (E1.1.9)**: `tests/red_team/test_broker_rejects_privileged_publish.py` boots NATS with a **probe fixture** (`deploy/nats-authz-test.conf`) aligned to ADR 0001 §3 kernel-only publishes. It asserts coordinator, meta, planner, and neuro identities cannot publish `decision.*`, `code.decision.*`, `policy.update`/`policy.*.status`, or `cognitive.response.validated`; neuro **may** publish `policy.restrict` (documented ADR §3 conflict / emergency halt path). This is not a drop-in production broker matrix. Application-layer signing is covered separately in `sdk/tests/test_signing.py`.

## Secrets Management

- All credentials go in `.env` (git-ignored), never in source files
- Production secrets are stored as GitHub Actions repository secrets
- Server access is restricted to authorized personnel via SSH keys
- NATS tokens should be rotated periodically

## Scope

This policy covers the `engramai/engram` repository. For issues in the SDK (`engramai/engram-sdk`), use the same reporting process.
