<!--
Thanks for contributing to Engram! Please fill out this template so reviewers
can understand and verify your change. Keep PRs focused — one feature or fix.
-->

## Summary

<!-- What does this PR do, and why? Link any related issue: "Closes #123". -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Documentation only
- [ ] Refactor / tooling / CI

## Component(s) affected

<!-- e.g. neuromorphic, sdk, dashboard, sensory-gateway, kernel, docs -->

## How was this tested?

<!-- Commands you ran and their result. For neuromorphic changes, include:
     cd neuromorphic && uv run python -m pytest tests/ -v -p no:anchorpy -->

## Checklist

- [ ] My code follows the project style (see `CONTRIBUTING.md`).
- [ ] I ran the relevant tests locally and they pass.
- [ ] I added or updated tests for my change (required for neuromorphic code).
- [ ] I updated documentation where needed.
- [ ] I did **not** commit secrets, credentials, server IPs, or large binaries.

## Architecture / safety alignment

<!-- Required for changes to the neuromorphic core or the safety layer. -->

- [ ] This change preserves the 6 architecture invariants (see `DESIGN-PRINCIPLES.md`). N/A if not touching `neuromorphic/`.
- [ ] This change does **not** weaken the safety layer (`kernel/`, `safety-supervisor/`, `beliefs/`). N/A if not touching those.
