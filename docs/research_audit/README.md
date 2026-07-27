# Factor-Gated Routing — Research Audit

**Audit version**: v2
**Audit commit**: aa14213af5667157000c94776fdd1d086575130e
**Code reference**: c6cc0968ccf4b39e6400792b6bdd38a4e57135cd
**Cutoff date**: 2026-07-27
**Last verified**: 2026-07-27
**Status**: COMPLETE (implementation BLOCKED pending specification gate)

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
| 15 | [Traceability & Compliance](15_TRACEABILITY_AND_COMPLIANCE.md) | Finding→task→test→gate→claim linkage matrix | COMPLETE |
| — | [audit_manifest.yaml](audit_manifest.yaml) | Machine-readable audit state with all counts | COMPLETE |

## Recommended Reading Order

1. Start with `00_EXECUTIVE_VERDICT.md` for the three verdicts
2. `01_REPOSITORY_FORENSICS.md` for what's actually in the codebase
3. `02_CLAIM_EVIDENCE_MATRIX.md` for what's claimed vs what's supported
4. `04_THEORY_REFORMULATION.md` for mathematical corrections
5. `05_ARCHITECTURE_OPTIONS.md` → `06_SELECTED_ARCHITECTURE_SPEC.md` for design
6. `10_IMPLEMENTATION_BACKLOG.md` for what to build
7. `11_TEST_AND_VERIFICATION_PLAN.md` for what to verify
8. `14_DEFINITION_OF_DONE.md` for when to declare readiness

## Key Decisions

- **Research direction**: CONDITIONAL GO (requires architecture pivot)
- **Primary architecture**: Candidate B (shared trunk + factor adapters)
- **Fallback**: Candidate A (fully independent streams)
- **Intervention semantics**: 8 canonical modes, typed InterventionSpec
- **Graph types**: INDEPENDENT, DAG, DENSE_DIRECTED, CUSTOM (DAG ≠ dense)
- **Evaluation**: paired-noise (NOT "counterfactual"), NoiseTrace via counter-seeds
- **Baselines**: CoInD→IndependentStreamDiT, EncDiff→CrossAttnDiT (rename)
- **Theory**: Path Non-Interference Theorem, Grönwall bound (sketch), NO monotonicity claim
- **Implementation**: BLOCKED until specification gate passed

## Blocked Items

1. Literature search saturation (API rate-limited; deferred to PI)
2. Implementation start (blocked by specification freeze)
3. GPU experiments (blocked by CPU property tests)
