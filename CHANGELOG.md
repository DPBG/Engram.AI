# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-source community health files: issue templates, pull request template,
  and a continuous integration workflow for the neuromorphic test suite.
- `CHANGELOG.md` (this file).

### Changed
- **Relicensed the project from proprietary to the MIT License.** Engram is now
  fully open source.
- Removed "closed-source / proprietary" markings from the meta-programmer
  subsystem; the entire stack is now released under MIT.
- Refreshed `README.md` with a project-status notice, known limitations, and
  status badges. Updated `CONTRIBUTING.md` for an open-source workflow.

### Security
- **Meta-Programmer source scanner no longer fails open on `from`-imported
  sinks (issue #118).** `meta_programmer.safety.scan_source` now resolves
  dangerous-sink calls through their import bindings, so command-execution
  sinks reached via `from os import system`, aliased imports
  (`from os import popen as p`, `import os as o`), and wholesale
  `from os import *` are flagged at **high** severity and blocked by the deploy
  gate — previously they produced only a *medium* `dangerous_import` finding and
  bypassed the high-severity-only gate.

## [0.1.0] - 2026-06-06

Initial public release of Engram — a self-aware, continuously-learning
neuromorphic AI system.

### Added
- Neuromorphic cognitive core: ~1M-neuron spiking neural network with
  integrated multi-mechanism learning (STDP, eligibility traces, BCM
  metaplasticity, 4-channel neuromodulation, homeostatic scaling, R-STDP).
- Developmental critical periods (infant → toddler → juvenile → adolescent →
  mature) with experience-dependent adolescent entry.
- Multi-compartment dendritic processing and cross-modal binding.
- MuJoCo virtual-body motor feedback loop and sensorimotor learning.
- Safety governance pipeline: Kernel, Safety Supervisor, Beliefs graph,
  Coordinator, and Meta-Programmer.
- Python SDK (`activelearning`): BaseService, SensorPlugin, ActuatorPlugin,
  EventBus.
- Standalone FastAPI dashboard with real-time WebSocket streaming.
- Sensory gateway with sensor discovery and video training pipeline.
- Three.js brain visualization demos.
- Pure-Python launcher (`run.py`) and Docker Compose deployment.
