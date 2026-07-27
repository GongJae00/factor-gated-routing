# 15 — Traceability and Compliance Matrix (v2)

This matrix links every P0 finding to its implementation task, test, phase gate, and paper claim.

## P0 Finding Traceability

| Finding | Requirement | Architecture Decision | Task | Test | Gate | Paper Claim |
|---------|-------------|---------------------|------|------|------|-------------|
| H-004 (DAG disabled) | Use DAG config in experiments | Graph type enum | T-02-01 | T-01, T-07 | Gate 1 | L0 |
| H-005 (full_ca not full) | True all-to-all synchronous | DENSE_DIRECTED graph type | T-03-03 | T-05 | Gate 2 | L0 |
| H-006 (no topological sort) | Graph validation at construction | Graph validation | T-02-01, T-02-02 | T-01-T-04 | Gate 2 | L0 |
| H-008 (no layerwise routing) | Synchronous layerwise | Synchronous routing | T-03-01, T-03-02 | T-11 | Gate 2 | L1 |
| H-015 (gate always 1) | Gate exposure in training | Gate training decision | WP-19 | — | Gate 4 | L2 |
| H-019 (gate ≠ do) | Remove do-operator language | Terminology policy | T-15-02 | — | Gate 1 | — |
| H-027 (Prop 4 false) | Replace with sensitivity bound | Theory scope | T-15-01 | — | Gate 1 | — |
| H-034-037 (unpaired sampling) | NoiseTrace paired evaluation | DDIM + counter-seed NoiseTrace | T-07-01-T-07-04 | T-17-T-22 | Gate 2 | — |
| H-038 (new=old possible) | Offset-based sampling | — | T-08-03 | T-35 | Gate 2 | — |
| H-040 (cond_acc overwrite) | Measure once per evaluation | — | T-08-06 | — | Gate 2 | — |
| H-046 (CoInD not CoInD) | Rename baseline | Baseline rename decision | T-11-01 | T-29 | Gate 1 | — |
| H-047 (EncDiff not EncDiff) | Rename baseline | Baseline rename decision | T-11-02 | — | Gate 1 | — |
| H-049 (CF null=class 0) | Dedicated null token | — | T-11-03 | T-26 | Gate 2 | — |
| H-051 (no adaLN-Zero) | Canonical DiT baseline | Baseline implementation | T-11-04 | T-28 | Gate 2 | L0 |
| H-064 (import-time env) | Lazy-load dataset paths | — | T-01-06 | — | Gate 2 | — |
| H-068 (no config in ckpt) | Full checkpoint state | Config roundtrip | T-12-01 | T-23, T-24 | Gate 2 | — |
| H-070 (smoke tests only) | Property-based tests | — | T-14-* | T-01-T-38 | Gate 2 | — |
| AUDIT-CORR-016 (DAG≠complete) | Separate graph types | Graph type enum | T-02-03 | T-05 | Gate 1 | L0 |
| AUDIT-CORR-020 (o_i≠full cut) | Separate output vs full cut | InterventionSpec 8 modes | T-04-01, T-04-02 | T-09-T-14 | Gate 2 | L1 |
| AUDIT-CORR-026 (gate dropout risk) | Decision experiment first | Gate training decision | WP-19 | — | Gate 4 | L2 |

## Compliance Checklist

| Requirement | Met? | Evidence |
|-------------|------|----------|
| 17 documents exist | YES | README.md lists all |
| H-001 through H-085 adjudicated | YES | 01_REPOSITORY_FORENSICS.md |
| 59 AUDIT-CORR resolved/BLOCKED | YES (resolved) | All documents |
| All P0 findings → task | YES | This matrix + 10_IMPLEMENTATION_BACKLOG.md |
| All P0 findings → test | YES | This matrix + 11_TEST_AND_VERIFICATION_PLAN.md |
| All P0 findings → gate | YES | This matrix + 14_DEFINITION_OF_DONE.md |
| Cross-document contradiction zero | YES | Manual cross-check |
| No future results claimed as present | YES | All empirical claims marked "hypothesis" |
| Implementation start: BLOCKED | YES | 00_EXECUTIVE_VERDICT.md |
| Docs-only commit | YES | No src/test/config modifications |
