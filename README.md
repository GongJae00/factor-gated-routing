# Gauge-Sensitive Inverse Generation

This repository is a clean public workspace for research on generative priors
for inverse problems.

## Research Question

When two generative dynamics produce comparable unconditional samples, do their
hidden path, gauge, or flux differences lead to different behavior under the
same inverse-conditioning rule?

The current claim is not that gauge freedom, flow matching, diffusion posterior
sampling, or inverse-problem solvers are new. The target is a controlled test of
conditional non-equivalence:

> unconditional equivalence does not necessarily imply conditional equivalence.

## Current Status

- Stage: research design.
- Public implementation: not started.
- Experiments: not started.
- Manuscript: not started.
- First allowed code scope: the E0-E2 synthetic falsification suite after the
  private design gate is accepted.

## Public Repo Policy

This repository should stay clean enough for public release:

- no local manuscript drafts;
- no private notes;
- no generated reports;
- no checkpoints, datasets, caches, or large artifacts;
- no debug notebooks unless intentionally curated;
- portable defaults before local GPU optimizations.

Private planning, PaperOrchestra inputs, and ledgers live outside this public
repository under GongJae's local research workspace.
