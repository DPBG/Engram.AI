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

<!--
Paste the commands you ran and their result. Full guide: docs/TESTING.md.
The suites CI runs:

  # Neuromorphic (primary)
  cd neuromorphic && uv run --extra dev python -m pytest tests/ -v -p no:anchorpy
  # SDK
  cd sdk && uv run --extra dev python -m pytest tests/ -v

Add or update tests that cover this change (required for neuromorphic code).
-->

## Checklist

- [ ] I ran the relevant tests locally and they pass (see `docs/TESTING.md`).
- [ ] I added or updated tests for my change (required for neuromorphic code).
- [ ] Lint / format / type gates pass locally — `ruff check .`, `black --check --line-length 100 .`, `mypy` (or `pre-commit run --all-files`).
- [ ] I addressed the automated **Ruff + Mypy** review comments left on the diff.
- [ ] My change follows the project style (see `CONTRIBUTING.md`).
- [ ] I updated documentation where needed.
- [ ] I did **not** commit secrets, credentials, server IPs, or large binaries.

## Architecture / safety alignment

<!-- Required for changes to the neuromorphic core or the safety layer. -->

- [ ] This change preserves the 6 architecture invariants (see `DESIGN-PRINCIPLES.md`). N/A if not touching `neuromorphic/`.
- [ ] This change does **not** weaken the safety layer (`kernel/`, `safety-supervisor/`, `beliefs/`). N/A if not touching those.

<!--
Before merge: the `ci-success` check must be green — it aggregates the
test / lint / type gates. CodeQL and the Ruff + Mypy review bot also post
findings directly on the diff; please resolve them.
-->
