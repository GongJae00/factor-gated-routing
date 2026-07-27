# Factor-Gated Routing — Research Audit

**Audit version**: v3
**Audit v2 commit**: aa14213af5667157000c94776fdd1d086575130e
**Audit v3 (spec freeze base)**: c700d341eb543c83e7d10ced034ffc2d8a179762
**Code reference**: c6cc0968ccf4b39e6400792b6bdd38a4e57135cd
**Cutoff date**: 2026-07-27
**Last verified**: 2026-07-27
**Spec version**: 3.0
**Status**: SPECIFICATION FROZEN (implementation UNBLOCKED for Phase 0 only)

---

## Purpose

This directory contains the complete pre-experiment master audit of the Factor-Gated Routing (FGR) research project. It evaluates code correctness (c6cc096), theoretical validity, novelty, experimental design, baseline fidelity, and publication readiness. No GPU experiments have been run; all findings are from static analysis.

## Document Index

| # | Document | Purpose | Status |
|---|----------|---------|--------|
| 00 | [Executive Verdict](00_EXECUTIVE_VERDICT.md) | Three verdicts (direction, audit completeness, implementation start) | COMPLETE |
| 01 | [Repository Forensics](01_REPOSITORY_FORENSICS.md) | Full file inventory, H-001 through H-085 adjudication | COMPLETE |
| 02 | [Claim-Evidence Matrix](02_CLAIM_EVIDENCE_MATRIX.md) | README+MATH_NOTES claims traced to code | COMPLETE |
| 03 | [Literature & Novelty Map](03_LITERATURE_NOVELTY_MAP.md) | Nearest works, overlap analysis, novelty verdict | BLOCKED (search not saturated) |
| 04 | [Theory Reformulation](04_THEORY_REFORMULATION.md) | Path Non-Interference Theorem, Grönwall bound, identifiability analysis | COMPLETE |
| 05 | [Architecture Options](05_ARCHITECTURE_OPTIONS.md) | 4 candidates compared, primary (B) + fallback (A) selected | COMPLETE |
| 06 | [Selected Architecture Spec](06_SELECTED_ARCHITECTURE_SPEC.md) | Canonical specification: graph types, InterventionSpec, data flow | COMPLETE |
| 07 | [Baseline Fidelity Plan](07_BASELINE_FIDELITY_PLAN.md) | Baseline rename/implement decisions, fairness regime | COMPLETE |
| 08 | [Data & Evaluation Protocol](08_DATA_AND_EVALUATION_PROTOCOL.md) | Dataset roles, splits, NoiseTrace, metrics, statistical plan | COMPLETE |
| 09 | [Experiment Master Matrix](09_EXPERIMENT_MASTER_MATRIX.md) | 31 experiment rows across 7 stages with success/failure gates | COMPLETE |
| 10 | [Implementation Backlog](10_IMPLEMENTATION_BACKLOG.md) | 60+ tasks across 19 work packages with dependency DAG | COMPLETE |
| 11 | [Test & Verification Plan](11_TEST_AND_VERIFICATION_PLAN.md) | 38 tests with acceptance criteria per dtype | COMPLETE |
| 12 | [Risk & Pivot Register](12_RISK_AND_PIVOT_REGISTER.md) | 23 risks with mitigation, 3 full pivot descriptions | COMPLETE |
| 13 | [Paper Positioning](13_PAPER_POSITIONING.md) | Title, claims ladder, paper structure, figures/tables | COMPLETE |
| 14 | [Definition of Done](14_DEFINITION_OF_DONE.md) | 8-phase gate checklist with kill criteria | COMPLETE |
| 15 | [Traceability & Compliance](15_TRACEABILITY_AND_COMPLIANCE.md) | Finding-to-task-to-test-to-gate-to-claim linkage matrix | COMPLETE |
| 16 | [Specification Freeze](16_SPECIFICATION_FREEZE.md) | Frozen spec v3.0: architecture, interventions, graphs, metrics, baselines, statistics | FROZEN |
| — | [audit_manifest.yaml](audit_manifest.yaml) | Machine-readable audit state with all counts | COMPLETE |
| — | [CORRECTION_REGISTER_V3.md](CORRECTION_REGISTER_V3.md) | 25 resolved corrections across v1→v3 audit versions | COMPLETE |

## Recommended Reading Order

1. Start with `00_EXECUTIVE_VERDICT.md` for the three verdicts
2. `01_REPOSITORY_FORENSICS.md` for what's actually in the codebase
3. `02_CLAIM_EVIDENCE_MATRIX.md` for what's claimed vs what's supported
4. `04_THEORY_REFORMULATION.md` for mathematical corrections
5. `05_ARCHITECTURE_OPTIONS.md` → `06_SELECTED_ARCHITECTURE_SPEC.md` for design
6. `10_IMPLEMENTATION_BACKLOG.md` for what to build
7. `11_TEST_AND_VERIFICATION_PLAN.md` for what to verify
8. `14_DEFINITION_OF_DONE.md` for when to declare readiness
9. `16_SPECIFICATION_FREEZE.md` for the frozen v3.0 spec and API contracts

## Key Decisions

- **Research direction**: CONDITIONAL GO
- **Primary architecture**: ROST-FRG (Read-Only Shared Trunk + Factor Residual Graph)
- **Fallback**: Fully Independent Additive Score Experts (Candidate A)
- **Intervention semantics**: 8 canonical modes, typed InterventionSpec
- **Graph types**: INDEPENDENT, DAG, DENSE_DIRECTED, CUSTOM_DIRECTED (DENSE_DIRECTED is not a DAG)
- **Evaluation**: paired-noise (NOT "counterfactual"), NoiseTrace via counter-seeds
- **Baselines**: CanonicalDiT (adalN-Zero), IndependentStreamDiT (formerly CoInD), CrossAttnDiT (formerly EncDiff), AllToAllFactorStreamDiT (formerly MMDiT-k), CF-DiT
- **Theory**: Path Non-Interference Theorem, Grönwall bound (sketch), NO monotonicity claim
- **Implementation**: Phase-0 UNBLOCKED; full implementation BLOCKED pending CPU property tests

## Blocked Items

1. Literature search saturation (API rate-limited; deferred to PI)
2. Full implementation start (blocked by CPU property tests; Phase-0 unblocked)
3. GPU experiments (blocked by CPU property tests)

## Specification Freeze

The v3.0 specification freeze is documented in `16_SPECIFICATION_FREEZE.md` and validated via `tools/validate_spec.py`. All 25 corrections in `CORRECTION_REGISTER_V3.md` are resolved. No open blocking questions at the specification level.
