# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it privately by emailing **rio@engram.ai**. Do not open a public issue.

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Security Architecture

- **Kernel**: Immutable safety gate. All motor commands and code proposals pass through the Kernel before execution. Cannot be modified or bypassed.
- **Safety Supervisor**: Risk analysis layer. Evaluates proposals for dangerous patterns.
- **Sandbox**: Ephemeral containers with no network, read-only filesystem, and resource limits for testing generated code.
- **NATS Authentication**: Production deployments use token-based auth. Tokens are stored in `.env` (git-ignored) and GitHub Actions secrets.

## Secrets Management

- All credentials go in `.env` (git-ignored), never in source files
- Production secrets are stored as GitHub Actions repository secrets
- Server access is restricted to authorized personnel via SSH keys
- NATS tokens should be rotated periodically

## Scope

This policy covers the `engramai/engram` repository. For issues in the SDK (`engramai/engram-sdk`), use the same reporting process.
