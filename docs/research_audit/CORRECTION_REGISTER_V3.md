# Correction Register v3

**Version**: 3.0
**Last updated**: 2026-07-27
**Total entries**: 25
**Status**: ALL RESOLVED

This register tracks every major correction applied across audit versions v1, v2, and v3.
Each entry links the original defect to its resolution and specifies a validation rule.

---

### CORR-001: AGENTS.md Hallucination — File Not at c6cc096

- **Source document**: `docs/research_audit/01_REPOSITORY_FORENSICS.md:L149` (H-085), `docs/research_audit/02_CLAIM_EVIDENCE_MATRIX.md`
- **Original text/claim**: AGENTS.md listed in file inventory as existing at commit c6cc096, with content note about `fgr/` package path reference.
- **Problem**: `git show c6cc096:AGENTS.md` confirms the file did NOT exist at the audited commit. The audit's file inventory included a non-existent file at the reference commit. The file now exists at HEAD with different content, but it was not part of the codebase at c6cc096.
- **Severity**: P1
- **Resolution**: Removed AGENTS.md from the file inventory in `01_REPOSITORY_FORENSICS.md` (code inventory table). H-085 re-adjudicated as PARTIALLY_VERIFIED (file exists at HEAD but was absent at c6cc096). Audit manifest no longer references AGENTS.md as a tracked source file.
- **Affected documents**: `docs/research_audit/01_REPOSITORY_FORENSICS.md`, `docs/research_audit/02_CLAIM_EVIDENCE_MATRIX.md`, `docs/research_audit/audit_manifest.yaml`
- **Validation rule**: `git show c6cc0968ccf4b39e6400792b6bdd38a4e57135cd:AGENTS.md` should return error (file not found). Audit inventory must not include AGENTS.md.
- **Status**: RESOLVED

### CORR-002: H-066 Status — PARTIALLY_VERIFIED (Mismatch Exists but Reason Differs)

- **Source document**: `docs/research_audit/01_REPOSITORY_FORENSICS.md:L125` (H-066)
- **Original text/claim**: "mixed_precision dtype mismatch" — originally claimed as a simple type mismatch without specifying the mechanism.
- **Problem**: The mismatch exists, but the original audit identified the wrong root cause. The actual mechanism: `src/train.py:L165` uses `torch.amp.autocast("cuda")` which defaults to fp16 dtype, while `TrainConfig.mixed_precision="bf16"` is declared in `src/config.py` but the TrainConfig class is never instantiated by `train.py`. So the configured dtype (bf16) and the default autocast dtype (fp16) diverge, but not through the path originally described.
- **Severity**: P1
- **Resolution**: H-066 re-adjudicated as PARTIALLY_VERIFIED with corrected evidence in `01_REPOSITORY_FORENSICS.md`. The mismatch pattern is confirmed via code inspection but the mechanism documentation is corrected.
- **Affected documents**: `docs/research_audit/01_REPOSITORY_FORENSICS.md`, `docs/research_audit/15_TRACEABILITY_AND_COMPLIANCE.md`
- **Validation rule**: Inspect `src/train.py:L165` for `torch.amp.autocast("cuda")` usage. Verify that `TrainConfig` in `src/config.py` is never instantiated in `train.py`. Confirm the two dtype settings (fp16 default vs bf16 configured) diverge.
- **Status**: RESOLVED

### CORR-003: H-073 Status — VERIFIED (.gitignore outputs/ vs default output/)

- **Source document**: `docs/research_audit/01_REPOSITORY_FORENSICS.md:L132` (H-073)
- **Original text/claim**: ".gitignore vs output dir mismatch" — originally marked as PARTIALLY_VERIFIED at confidence 0.80, with uncertain root cause attribution.
- **Problem**: The audit initially hesitated on full VERIFIED status because the mismatch mechanism was unclear. Upon rerun at c6cc096: `.gitignore` contains the pattern `outputs/` (plural) at line 30, while the default output directory used by training scripts is `output/` (singular). The mismatch is exact and reproducible.
- **Severity**: P2
- **Resolution**: H-073 promoted from PARTIALLY_VERIFIED (0.80 confidence) to VERIFIED (0.95 confidence). Evidence clarified: `.gitignore` ignores `outputs/` (plural) but default output is `output/` (singular). Either the gitignore pattern needs updating or the output directory name needs changing.
- **Affected documents**: `docs/research_audit/01_REPOSITORY_FORENSICS.md`
- **Validation rule**: `rg "output" .gitignore` at c6cc096 shows `outputs/`. Default output path in train.py references `output/` (singular). Confirm they don't match.
- **Status**: RESOLVED

### CORR-004: H-076 through H-085 Individual Adjudication — Each Has Separate Entry

- **Source document**: `docs/research_audit/01_REPOSITORY_FORENSICS.md:L136-149`
- **Original text/claim**: H-076 through H-085 were originally presented as a block group "New defects (H-076 through H-085 from v1 audit)" with a subheading but not individually triaged in the initial audit pass.
- **Problem**: Grouping 10 distinct hypotheses under a single heading without individual P0/P1/P2 severity classification and individual remediation tracking makes it impossible to track which defects are resolved and which remain open.
- **Severity**: P1
- **Resolution**: Each hypothesis (H-076 through H-085) now has its own row in the adjudication table with individual status, confidence, and evidence. H-076 (stale README path) VERIFIED. H-077 (build_config hardcodes use_gating) VERIFIED. H-078 (factor_idx no validation) VERIFIED. H-079 (factor_names bug) VERIFIED. H-080 (new_val == old_val) VERIFIED. H-081 (RuntimeError on unexpected keys) VERIFIED. H-082 (no adaLN-Zero) VERIFIED. H-083 (FGRConfig never instantiated) VERIFIED. H-084 (cosine schedule correct) VERIFIED. H-085 (AGENTS.md not at c6cc096) PARTIALLY_VERIFIED.
- **Affected documents**: `docs/research_audit/01_REPOSITORY_FORENSICS.md`
- **Validation rule**: Count H-076 through H-085 entries in the adjudication table — must be exactly 10 rows with individual status values.
- **Status**: RESOLVED

### CORR-005: 3DShapes Component Independence — Factors Are Independent, Not Correlated

- **Source document**: `docs/research_audit/01_REPOSITORY_FORENSICS.md:L110` (H-058), `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md:L8`
- **Original text/claim**: Early audit notes suggested 3DShapes factors might be correlated or that DAG relationships could exist between them.
- **Problem**: 3DShapes factors are generated independently via uniform Cartesian product — all 480,000 combinations exist exactly once. There is no generative process correlation, no SCM between factors. Any DAG imposed on 3DShapes factors is an architectural hypothesis, not a ground-truth relationship. Claiming correlation or causal dependence would be factually incorrect.
- **Severity**: P1
- **Resolution**: All documents updated to state "3DShapes factors are independent" — the dataset is a uniform Cartesian product. The primary graph type for 3DShapes is INDEPENDENT. Any DAG experiment on 3DShapes is explicitly labeled as testing an architectural hypothesis, not a ground-truth causal structure.
- **Affected documents**: `docs/research_audit/01_REPOSITORY_FORENSICS.md`, `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md`, `docs/research_audit/spec/graphs.yaml`
- **Validation rule**: `rg -i "3dshapes.*correlat" docs/research_audit/` should return zero matches. `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md` must state "all 480,000 Cartesian combinations exist once" or equivalent independence assertion.
- **Status**: RESOLVED

### CORR-006: 3DShapes Raw Size — ~5.49 GB uint8, Not ~2.4 GB

- **Source document**: `docs/research_audit/01_REPOSITORY_FORENSICS.md:L115` (H-059), `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md:L102`
- **Original text/claim**: 3DShapes full RAM load estimated at "~22 GB for 480K×64×64×3 float32" in H-059, with indirect implications that the raw storage was ~2.4 GB.
- **Problem**: The raw 3DShapes dataset on disk is stored as uint8. Calculation: 480,000 × 64 × 64 × 3 × 1 byte = 5,898,240,000 bytes ≈ 5.49 GB. The ~2.4 GB figure was approximately half of the correct value. The ~22 GB figure (float32) is correct for in-memory representation after conversion, but the on-disk size was understated.
- **Severity**: P2
- **Resolution**: Corrected to "3DShapes provides `.h5` with `images` (480,000, 64, 64, 3) and factor labels. Raw uint8 size: ~5.49 GB on disk. Float32 in-memory: ~22 GB." This correction appears in `08_DATA_AND_EVALUATION_PROTOCOL.md:L102`.
- **Affected documents**: `docs/research_audit/01_REPOSITORY_FORENSICS.md`, `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md`
- **Validation rule**: Search `docs/research_audit/` for "5.49 GB" — must appear in 3DShapes context. Search for "2.4 GB" — must NOT appear in 3DShapes context.
- **Status**: RESOLVED

### CORR-007: Architecture 05 vs 06 Conflict — Unified to ROST-FRG

- **Source document**: `docs/research_audit/05_ARCHITECTURE_OPTIONS.md:L22-33`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md:L1-6`
- **Original text/claim**: Document 05 used "Candidate B: Shared Factor-Agnostic Trunk + Factor Adapters" while document 06 used "ROST-FRG (Read-Only Shared Trunk + Factor Residual Graph)". The names disagreed and readers could not tell if they referred to the same architecture.
- **Problem**: Document 05 referred to the primary architecture by its candidate label ("Candidate B") without using the canonical name "ROST-FRG". Document 06 introduced "ROST-FRG" but did not explicitly state it is the same thing as Candidate B. This created a name collision across documents.
- **Severity**: P2
- **Resolution**: All documents unified to "ROST-FRG (Read-Only Shared Trunk + Factor Residual Graph)". Document 05 header now says "### B. ROST-FRG — Read-Only Shared Trunk + Factor Residual Graph" with explicit cross-reference to 06. Document 06 uses "ROST-FRG" as primary name. `spec/ARCHITECTURE_SPEC.md:L1` uses ROST-FRG as canonical. `audit_manifest.yaml:L96` shows primary_architecture: ROST-FRG.
- **Affected documents**: `docs/research_audit/05_ARCHITECTURE_OPTIONS.md`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md`, `docs/research_audit/spec/ARCHITECTURE_SPEC.md`, `docs/research_audit/spec/architecture.yaml`, `docs/research_audit/audit_manifest.yaml`
- **Validation rule**: `rg "Candidate B" docs/research_audit/` should return only entries that include "ROST-FRG" in the same line or explicitly note the equivalence (description header lines). No document may use "Candidate B" as a standalone architecture name without the ROST-FRG qualifier.
- **Status**: RESOLVED

### CORR-008: Intervention Mode Names — 6 to 8 Canonical Modes, Stale Names Banned

- **Source document**: `docs/research_audit/spec/INTERVENTION_SPEC.md:L12-21`
- **Original text/claim**: Early audit versions listed only 6 intervention modes: observational, factor_edit, direct_output_ablation, node_deletion, edge_ablation, and one conflated "path_ablation/full_source_cut" mode. The stale names `path_ablation`, `full_source_cut`, `graph_surgery`, `do_like`, `output_gate_only`, `drop_factor`, `zero_out`, `intervene` were used interchangeably in code and documentation.
- **Problem**: 6 modes conflated two distinct operations (source cut vs output ablation). Stale names created ambiguity and prevented automated verification. "path_ablation" was simultaneously used for three different semantics across files. This blocked the specification freeze gate.
- **Severity**: P0
- **Resolution**: Canonical set expanded to 8 modes: OBSERVATIONAL, FACTOR_EDIT, CONDITION_MASK, DIRECT_OUTPUT_ABLATION, EDGE_ABLATION, NODE_DELETION, FACTOR_SOURCE_CUT, NEURAL_GRAPH_SURGERY. `spec/INTERVENTION_SPEC.md` is the single source of truth for all mode names and semantics. Stale names deprecated with explicit replacement rules in the "Stale Names" table. Cross-document consistency declared (`spec/status.yaml` cross_document_consistency). Mode enum frozen at 8 — no additions or deletions without a spec amendment proposal.
- **Affected documents**: `docs/research_audit/spec/INTERVENTION_SPEC.md`, `docs/research_audit/spec/architecture.yaml`, `docs/research_audit/spec/status.yaml`, `docs/research_audit/00_EXECUTIVE_VERDICT.md`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md`, `docs/research_audit/13_PAPER_POSITIONING.md`, code files under `src/`, test files under `tests/`
- **Validation rule**: `rg -i "path_ablation|full_source_cut|do_like|do-like|output_gate_only|drop_factor" spec/INTERVENTION_SPEC.md` — these may appear ONLY in the "Stale Names" deprecation table. `rg "class InterventionMode" src/` must show exactly 8 enum members matching the canonical names.
- **Status**: RESOLVED

### CORR-009: Factor Source Cut vs Output Ablation — Now Separated

- **Source document**: `docs/research_audit/spec/INTERVENTION_SPEC.md:L30-34`
- **Original text/claim**: Early audit documents treated "output gate zero" and "source gate zero" as equivalent or interchangeable for invariance testing. Some code paths used `output_gate=0` to test invariance, conflating output silencing with path cutting.
- **Problem**: Setting `output_gate_i=0` silences factor i's additive output contribution, but does NOT prevent factor i from influencing other branches through edge messages. The branch still computes with trunk + source + edge inputs; only the final output head is zeroed. This is insufficient to guarantee Path Non-Interference. True invariance requires `source_gate_i=0` (FACTOR_SOURCE_CUT), which prevents the factor value from ever entering the computation graph.
- **Severity**: P0
- **Resolution**: The two modes are now explicitly separated in `spec/INTERVENTION_SPEC.md`: DIRECT_OUTPUT_ABLATION (output_gate=0, source_gate=1, node stays active, edges preserved) — no invariance guarantee, descriptive only. FACTOR_SOURCE_CUT (source_gate=0, output_gate=1, node stays active, edges preserved) — Factor-Source Path Non-Interference theorem applies. The "output_gate=0 as invariance test" pattern is explicitly marked as incorrect in the invariance theorem scope note (`spec/INTERVENTION_SPEC.md:L150`).
- **Affected documents**: `docs/research_audit/spec/INTERVENTION_SPEC.md`, `docs/research_audit/04_THEORY_REFORMULATION.md`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md`
- **Validation rule**: `spec/INTERVENTION_SPEC.md` mode semantics table: DIRECT_OUTPUT_ABLATION row must state "no invariance guarantee". FACTOR_SOURCE_CUT row must reference "Factor-Source Path Non-Interference". No document may claim DIRECT_OUTPUT_ABLATION provides invariance.
- **Status**: RESOLVED

### CORR-010: "complete DAG" Terminology — DENSE_DIRECTED Is Not a DAG

- **Source document**: `docs/research_audit/spec/graphs.yaml:L56-57`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md:L43`
- **Original text/claim**: The phrase "complete DAG" was used in early audit documents to describe a graph where every factor connects to every other factor.
- **Problem**: DENSE_DIRECTED contains the edge set `{(j,i) | j != i}`. When K > 1, this includes 2-cycles (e.g., both (0,1) and (1,0) exist). A DAG cannot contain cycles by definition. Therefore "complete DAG" is a contradiction in terms. Calling DENSE_DIRECTED a "DAG" would confuse readers about whether cycles exist and whether topological order is available.
- **Severity**: P1
- **Resolution**: The phrase "complete DAG" is banned across all documents, code, and paper drafts. The correct name is "DENSE_DIRECTED". `spec/graphs.yaml` explicitly states: "DENSE_DIRECTED is NOT a DAG. The phrase 'complete DAG' is forbidden in documentation and code comments." The `forbidden_phrases` section in `graphs.yaml` lists both "complete DAG" and "fully connected DAG" with severity: error.
- **Affected documents**: `docs/research_audit/spec/graphs.yaml`, `docs/research_audit/spec/ARCHITECTURE_SPEC.md`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md`, `docs/research_audit/13_PAPER_POSITIONING.md`
- **Validation rule**: `rg -i "complete dag" docs/research_audit/` and `rg -i "fully connected dag" docs/research_audit/` must return zero results in non-deprecation contexts. May appear ONLY in forbidden-phrase tables or correction registers.
- **Status**: RESOLVED

### CORR-011: Output-Gate-Only Invariance — Replaced with FACTOR_SOURCE_CUT

- **Source document**: `docs/research_audit/04_THEORY_REFORMULATION.md:L69-81`, `docs/research_audit/spec/INTERVENTION_SPEC.md:L150`
- **Original text/claim**: Earlier versions implied that setting `output_gate_i=0` was sufficient to test factor-source invariance — that zeroing the output head meant the factor was "removed" from the computation.
- **Problem**: With `output_gate_i=0`, factor i's source embedding still enters the branch computation, propagates through edge messages to other branches, and influences their factor adapter states. Even though factor i's own output is zeroed, its influence persists in the system through cross-branch edges. This is a partial cut, not a complete cutset. The Path Non-Interference Theorem requires a complete cutset — all paths from the factor source to ANY output must be severed.
- **Severity**: P0
- **Resolution**: The invariance theorem's scope note (`spec/INTERVENTION_SPEC.md:L150-151`) explicitly states: "This theorem holds for FACTOR_SOURCE_CUT mode. It does NOT hold for DIRECT_OUTPUT_ABLATION." All invariance testing protocols reference FACTOR_SOURCE_CUT, not DIRECT_OUTPUT_ABLATION. The stale name `output_gate_only` is deprecated with replacement `FACTOR_SOURCE_CUT` in the stale names table.
- **Affected documents**: `docs/research_audit/spec/INTERVENTION_SPEC.md`, `docs/research_audit/04_THEORY_REFORMULATION.md`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md`, `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md`
- **Validation rule**: No document may claim that output_gate=0 alone provides factor-source invariance. Protocol 2 in `08_DATA_AND_EVALUATION_PROTOCOL.md` must use FACTOR_SOURCE_CUT mode. Invariance test code must manipulate `source_gate`, not `output_gate`.
- **Status**: RESOLVED

### CORR-012: "paired counterfactual" Terminology — Replaced with "paired-noise evaluation"

- **Source document**: `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md:L409`, `docs/research_audit/13_PAPER_POSITIONING.md:L13`
- **Original text/claim**: The evaluation protocol was described as "paired counterfactual evaluation" or referenced "counterfactual" sampling.
- **Problem**: "Counterfactual" has precise SCM semantics (Pearl's three-step procedure: abduction, action, prediction on a structural causal model). The FGR architecture does not implement an SCM and does not support Pearl-style counterfactual computation. Using "counterfactual" terminology without SCM justification is a category error and risks reviewer rejection.
- **Severity**: P1
- **Resolution**: All occurrences replaced with "paired-noise evaluation" or "common-random-number coupling". The mechanism is: same NoiseTrace (same x_T, same per-step noise), different factor conditions, compare outputs. This is a shared-random-number design, not a causal counterfactual. "Counterfactual" is permitted ONLY in literature/future-work contexts with explicit qualification. The terminology policy in `08_DATA_AND_EVALUATION_PROTOCOL.md:L407-417` codifies this.
- **Affected documents**: `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md`, `docs/research_audit/13_PAPER_POSITIONING.md`, `docs/research_audit/00_EXECUTIVE_VERDICT.md`, `docs/research_audit/04_THEORY_REFORMULATION.md`
- **Validation rule**: `rg -i "paired counterfactual" docs/research_audit/` must return zero results. `rg -i "counterfactual" docs/research_audit/` may return results only in literature context, terminology policy table, or correction registers.
- **Status**: RESOLVED

### CORR-013: "do-like" — Removed, Replaced with NEURAL_GRAPH_SURGERY

- **Source document**: `docs/research_audit/spec/INTERVENTION_SPEC.md:L135`
- **Original text/claim**: The term "do-like" or "do-like intervention" was used in code (`tests/test_fgr_model.py`) and early audit documents to describe an intervention mode that cuts incoming edges and injects a new factor value.
- **Problem**: "do" references Pearl's do-operator which performs an intervention on a structural causal model by setting a variable to a value and severing all incoming causal edges. FGR's graph surgery is a neural computation manipulation (cutting neural edges, injecting a new activation), not a causal intervention on an SCM. Using "do" terminology implies SCM semantics that are not justified, creating a paper-rejection risk.
- **Severity**: P1
- **Resolution**: "do-like" and "do-like intervention" are banned from all code, documentation, and paper drafts. The canonical replacement is "NEURAL_GRAPH_SURGERY". The stale names table (`spec/INTERVENTION_SPEC.md:L135`) lists `do_like / do-like → NEURAL_GRAPH_SURGERY` with reason: "no causal claim; do terminology is misleading".
- **Affected documents**: `docs/research_audit/spec/INTERVENTION_SPEC.md`, `docs/research_audit/04_THEORY_REFORMULATION.md`, `tests/test_fgr_model.py`, `tests/test_fgr_diffusion.py`
- **Validation rule**: `rg -i "do.like\|do-like" docs/research_audit/ tests/` must return zero results in non-deprecation contexts.
- **Status**: RESOLVED

### CORR-014: "graph_surgery" to NEURAL_GRAPH_SURGERY

- **Source document**: `docs/research_audit/spec/INTERVENTION_SPEC.md:L133`
- **Original text/claim**: The intervention mode was named "graph_surgery" in early audit documents and code.
- **Problem**: "graph_surgery" is ambiguous — it could refer to actual graph structure editing (modifying the edge set of the factor graph) rather than a neural intervention at inference time. The term "neural" qualifier distinguishes this from graph-structure edits. Additionally, "graph_surgery" without "neural" risks conflation with the causal inference literature's graph surgery operations.
- **Severity**: P2
- **Resolution**: Renamed to "NEURAL_GRAPH_SURGERY". The stale names table lists `graph_surgery → NEURAL_GRAPH_SURGERY` with reason: "missing neural qualifier; can be confused with graph edits". All spec documents, code enums, and intervention protocols use the canonical name.
- **Affected documents**: `docs/research_audit/spec/INTERVENTION_SPEC.md`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md`, `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md`, `docs/research_audit/13_PAPER_POSITIONING.md`
- **Validation rule**: `rg "graph_surgery" docs/research_audit/` must return results only in the stale names deprecation table or correction registers. `rg "NEURAL_GRAPH_SURGERY" docs/research_audit/` must appear as the canonical name.
- **Status**: RESOLVED

### CORR-015: "path_ablation" — Reified to FACTOR_SOURCE_CUT or DIRECT_OUTPUT_ABLATION

- **Source document**: `docs/research_audit/spec/INTERVENTION_SPEC.md:L131`
- **Original text/claim**: "path_ablation" was the primary intervention mode name in code (`tests/test_fgr_model.py`) and early audit documents, used to describe both source-gate-cutting and output-gate-zeroing.
- **Problem**: "path_ablation" is fatally ambiguous: it could mean (a) cutting the factor source gate (preventing factor value from entering), (b) cutting the output gate (silencing output contribution), (c) cutting edges (blocking inter-branch messages), or (d) all of the above. Without disambiguation, two researchers running "path_ablation" experiments could get different results depending on which gates they manipulated.
- **Severity**: P0
- **Resolution**: "path_ablation" is banned as a mode name. It is replaced by FACTOR_SOURCE_CUT when cutting the factor source gate, and by DIRECT_OUTPUT_ABLATION when zeroing the output contribution. In cases where the code originally used "path_ablation" with unclear semantics, the correct replacement is determined by inspecting which gates were actually manipulated. The stale names table (`spec/INTERVENTION_SPEC.md:L131`) documents the disambiguation rule.
- **Affected documents**: `docs/research_audit/spec/INTERVENTION_SPEC.md`, `tests/test_fgr_model.py`, `tests/test_fgr_diffusion.py`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md`
- **Validation rule**: `rg "path_ablation" docs/research_audit/ tests/ src/` must return results only in stale name deprecation tables or correction registers. No code path may use "path_ablation" as a mode identifier.
- **Status**: RESOLVED

### CORR-016: "child-before-parent ordering" in Tests — Removed, Sync Updates Do Not Care About Order

- **Source document**: `docs/research_audit/01_REPOSITORY_FORENSICS.md:L47` (H-006), `docs/research_audit/spec/ARCHITECTURE_SPEC.md:L36`
- **Original text/claim**: Some test descriptions and early audit analysis discussed "child-before-parent" ordering concerns, suggesting that the forward pass needed topological ordering to ensure parents are processed before children.
- **Problem**: Under the synchronous layerwise update protocol (canonical since spec v3.0), all branch states at layer l are computed from a frozen snapshot of layer l-1 branch states. There is no intra-layer sequential dependency — every branch reads from the same frozen snapshot. Therefore, the order in which branches are processed within a layer is irrelevant. "Child-before-parent" is not a meaningful concern; it conflates sequential update semantics with synchronous snapshot semantics.
- **Severity**: P2
- **Resolution**: "child-before-parent ordering" removed from test descriptions and implementation concerns. The synchronous update protocol is documented in `spec/ARCHITECTURE_SPEC.md:L36` ("layerwise synchronous: all branch states at layer l are computed from a frozen snapshot of layer l-1 branch states"). The graph validation performs topological sort for reporting and validation only, NOT for forward execution ordering (`spec/graphs.yaml:L30`).
- **Affected documents**: `docs/research_audit/01_REPOSITORY_FORENSICS.md`, `docs/research_audit/spec/ARCHITECTURE_SPEC.md`, `docs/research_audit/spec/graphs.yaml`, `docs/research_audit/10_IMPLEMENTATION_BACKLOG.md`
- **Validation rule**: `rg -i "child.before.parent" docs/research_audit/` must return zero results. `spec/ARCHITECTURE_SPEC.md` must state that updates are synchronous snapshot-based.
- **Status**: RESOLVED

### CORR-017: Chain DAG on dSprites Where Factors Are Independent — Fixed to INDEPENDENT

- **Source document**: `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md:L7-8`, `docs/research_audit/01_REPOSITORY_FORENSICS.md:L110` (H-058)
- **Original text/claim**: Some experiment descriptions implied or suggested using a chain DAG (e.g., shape→scale→rotation) on the dSprites dataset.
- **Problem**: dSprites factors (shape, scale, rotation) are generated independently. There is no ground-truth causal chain. Imposing a chain DAG on independent factors would be testing an arbitrary architectural hypothesis, not a meaningful causal relationship. Worse, it could mislead reviewers into thinking the ground-truth data has a causal structure. The correct graph type for dSprites is INDEPENDENT.
- **Severity**: P1
- **Resolution**: All experiment plans updated to use INDEPENDENT as the default graph type for dSprites. Any DAG experiment on dSprites must carry an explicit note that it tests architectural structure, not ground-truth causality. The graph type assignments in `08_DATA_AND_EVALUATION_PROTOCOL.md` are explicit: dSprites uses INDEPENDENT by default.
- **Affected documents**: `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md`, `docs/research_audit/01_REPOSITORY_FORENSICS.md`, `docs/research_audit/09_EXPERIMENT_MASTER_MATRIX.md`
- **Validation rule**: `rg "dsprites.*chain\|chain.*dsprites" docs/research_audit/` must return zero results in experiment design contexts. dSprites experiment descriptions must default to INDEPENDENT graph type.
- **Status**: RESOLVED

### CORR-018: p<0.05 with 2 Seeds in Pilot — Removed, Direction-Only

- **Source document**: `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md:L331-332`, `docs/research_audit/13_PAPER_POSITIONING.md`
- **Original text/claim**: Some evaluation descriptions implied or stated that statistical significance testing (p<0.05) would be performed on pilot results with 2 training seeds.
- **Problem**: A 2-seed pilot provides insufficient statistical power for any meaningful significance test. Minimum sample size for a two-sided Wilcoxon or binomial test at α=0.05 is typically 5 pairs for any detectable effect. With 2 seeds, even a perfect split (both seeds showing the same direction) only achieves p≈0.25 under the sign test (minimum attainable p = 2×(1/2)^2 = 0.5 for two seeds). Reporting p<0.05 from 2 seeds is a type-I error rate inflation and would not survive peer review.
- **Severity**: P1
- **Resolution**: Pilot phase (2-3 seeds) reports direction of effect, variance estimate, and effect size only — no p-values. Confirmatory phase (5 seeds) reports paired bootstrap 95% CI and binomial CI. `08_DATA_AND_EVALUATION_PROTOCOL.md:L325-334` explicitly separates pilot (direction only, no p-values) from confirmatory (full CIs). The claims ladder in `13_PAPER_POSITIONING.md` marks all L2-L3 claims as hypotheses pending experiment with no premature statistical claims.
- **Affected documents**: `docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md`, `docs/research_audit/13_PAPER_POSITIONING.md`, `docs/research_audit/14_DEFINITION_OF_DONE.md`
- **Validation rule**: `rg "p.*0\.05\|p.*0\.01\|p-value.*pilot\|pilot.*p-value" docs/research_audit/` must return zero results linking p-values to pilot experiments. Pilot sections must reference "direction of effect" and "variance estimate" only.
- **Status**: RESOLVED

### CORR-019: WP15-->all Mermaid Edge — Replaced with Actual Dependencies

- **Source document**: `docs/research_audit/10_IMPLEMENTATION_BACKLOG.md:L187-243`
- **Original text/claim**: In the v2 implementation backlog, the mermaid dependency diagram showed WP-15 (Documentation) with edges pointing to all other work packages, creating a false hub dependency where documentation blocked every other task.
- **Problem**: Documentation (WP-15) does not need to block graph validation (WP-02), intervention API (WP-04), or sampler implementation (WP-07). Having WP-15 as a prerequisite for all other WPs creates an artificial bottleneck. The actual dependency structure is: WP-15 depends on WP-01 (config unification), WP-02 (graph validation), and WP-04 (intervention API) because docs describe these components, but WP-15 does not block their implementation.
- **Severity**: P2
- **Resolution**: The mermaid dependency DAG in `10_IMPLEMENTATION_BACKLOG.md` now shows proper dependency structure. WP-15 is placed as a parallel workstream that reads from WP-01, WP-02, WP-04 (not the other way around). WP-15 has edges FROM (not TO) the packages it documents. No package has WP-15 as a prerequisite.
- **Affected documents**: `docs/research_audit/10_IMPLEMENTATION_BACKLOG.md`
- **Validation rule**: In the mermaid diagram in `10_IMPLEMENTATION_BACKLOG.md`, no edge should originate from WP15 and terminate at another WP. WP15 should appear only as a target of edges (from WPs it documents), not as a source.
- **Status**: RESOLVED

### CORR-020: "6 modes" to "8 modes" in Backlog and Paper Positioning

- **Source document**: `docs/research_audit/10_IMPLEMENTATION_BACKLOG.md`, `docs/research_audit/13_PAPER_POSITIONING.md`
- **Original text/claim**: Multiple documents referenced "6 intervention modes" reflecting the early audit's mode count before CONDITION_MASK and FACTOR_SOURCE_CUT were separated from other modes.
- **Problem**: The canonical InterventionMode enum has 8 members. Any document stating "6 modes" is stale and references a superseded version of the intervention taxonomy. This mismatch between the spec (8 modes) and descriptive documents (6 modes) was a cross-document consistency failure.
- **Severity**: P1
- **Resolution**: All occurrences of "6 modes" updated to "8 modes". The canonical count of 8 is established in `spec/INTERVENTION_SPEC.md` and `spec/architecture.yaml:L113` (`intervention_modes: 8`). References to "6 modes" are permitted only in historical context (describing audit v1 state) with explicit qualification.
- **Affected documents**: `docs/research_audit/10_IMPLEMENTATION_BACKLOG.md`, `docs/research_audit/13_PAPER_POSITIONING.md`, `docs/research_audit/00_EXECUTIVE_VERDICT.md`
- **Validation rule**: `rg "6 modes" docs/research_audit/` must return zero results in non-historical contexts. Documents referencing the current spec must state "8 modes".
- **Status**: RESOLVED

### CORR-021: "18 Work Packages" to "19 Work Packages"

- **Source document**: `docs/research_audit/10_IMPLEMENTATION_BACKLOG.md:L3`
- **Original text/claim**: The implementation backlog header stated "19 Work Packages" but the actual count of WP sections in the v2 document was 18. The document declared 19 but only enumerated 18.
- **Problem**: WP-19 ("Gate Training Decision") was referenced in the dependency DAG but had no corresponding section heading in the backlog document body. The mermaid diagram referenced WP19 as a node but the section list only went up to WP-18 (Reproducibility). This made the document internally inconsistent.
- **Severity**: P2
- **Resolution**: Now actually 19 work packages: WP-00 through WP-18 are explicitly listed in the backlog, and WP-19 is included as the dependency DAG node for the gate training decision experiment. The "Total tasks: 60+ across 19 Work Packages" header is correct. The mermaid diagram references all 19 packages.
- **Affected documents**: `docs/research_audit/10_IMPLEMENTATION_BACKLOG.md`
- **Validation rule**: Count WP sections in `10_IMPLEMENTATION_BACKLOG.md` — must be 19 (WP-00 through WP-18, plus WP-19 in the dependency DAG). The mermaid diagram node count must be 20 (including the WP labels).
- **Status**: RESOLVED

### CORR-022: Novelty Claim Locked to PROVISIONAL Until Literature Saturated

- **Source document**: `docs/research_audit/03_LITERATURE_NOVELTY_MAP.md`, `docs/research_audit/13_PAPER_POSITIONING.md`
- **Original text/claim**: Early audit documents asserted novelty of the approach (factor-path routing, 8-mode intervention interface, path non-interference theorem) as though it were a confirmed finding.
- **Problem**: Literature search is BLOCKED (API rate-limiting prevented exhaustive automated search). Without saturated literature review, novelty cannot be confirmed. Any novelty claim without literature verification is premature and risks being invalidated by prior work. Key collision risks include: DisDiff (factor score decomposition, NeurIPS 2023), CBDiffuse (concept bottleneck diffusion), GSDM (DAG-in-architecture, ICML 2023).
- **Severity**: P1
- **Resolution**: All novelty claims locked to PROVISIONAL with explicit qualification: "pending literature saturation." The recommendation is titled "Recommended Title (**pending literature verification**)" in `13_PAPER_POSITIONING.md:L5`. The claims ladder marks all L2-L3 claims as hypotheses. `spec/status.yaml:L32-40` records literature_validation as BLOCKED with documented reason.
- **Affected documents**: `docs/research_audit/03_LITERATURE_NOVELTY_MAP.md`, `docs/research_audit/13_PAPER_POSITIONING.md`, `docs/research_audit/spec/status.yaml`, `docs/research_audit/00_EXECUTIVE_VERDICT.md`
- **Validation rule**: No document may state novelty as a fact without the "pending literature verification" or "PROVISIONAL" qualifier. `spec/status.yaml` literature_validation must remain BLOCKED until resolved.
- **Status**: RESOLVED

### CORR-023: Architecture Renamed — Candidate B to ROST-FRG

- **Source document**: `docs/research_audit/05_ARCHITECTURE_OPTIONS.md:L22-33`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md:L3`
- **Original text/claim**: The primary architecture was referred to as "Candidate B" or "Architecture B" without a canonical descriptive name.
- **Problem**: "Candidate B" is a temporary evaluation label, not a suitable permanent name for a research artifact. It conveys no semantic information about the architecture. For paper writing, code references, and figure captions, a descriptive name is required. Continuing to use temporary candidate labels in frozen spec documents would confuse readers and dilute the branding of the contribution.
- **Severity**: P2
- **Resolution**: Standardized as "ROST-FRG (Read-Only Shared Trunk + Factor Residual Graph)". The acronym is pronounceable, the expansion is descriptive, and the canonical name is used in all spec files, architecture documents, and paper positioning materials. `spec/architecture.yaml:L2-3` defines the canonical name. `spec/ARCHITECTURE_SPEC.md:L1` uses ROST-FRG as the title. "Candidate B" is retained only in `05_ARCHITECTURE_OPTIONS.md` as the historical label during the selection process, with explicit cross-reference to ROST-FRG.
- **Affected documents**: `docs/research_audit/05_ARCHITECTURE_OPTIONS.md`, `docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md`, `docs/research_audit/spec/ARCHITECTURE_SPEC.md`, `docs/research_audit/spec/architecture.yaml`, `docs/research_audit/audit_manifest.yaml`, `docs/research_audit/00_EXECUTIVE_VERDICT.md`
- **Validation rule**: `rg "Candidate B" docs/research_audit/` must return results only in `05_ARCHITECTURE_OPTIONS.md` (historical context) with ROST-FRG adjacency. All other documents must use ROST-FRG as the canonical name.
- **Status**: RESOLVED

### CORR-024: MMDiT-k Baseline Name — Renamed to AllToAllFactorStreamDiT

- **Source document**: `docs/research_audit/07_BASELINE_FIDELITY_PLAN.md:L34-43`, `docs/research_audit/10_IMPLEMENTATION_BACKLOG.md`
- **Original text/claim**: The baseline was named "MMDiT-k" where the "-k" suffix signaled a K-stream generalization of the 2-stream MMDiT architecture from Stable Diffusion 3.
- **Problem**: The name "MMDiT-k" implies it IS MMDiT with more streams. But FGR's all-to-all stream architecture differs from MMDiT in several key ways: (a) MMDiT uses separate text and image token streams with designated joint attention layers; (b) FGR has K factor streams without the text/image modality distinction; (c) the attention pattern (all-to-all vs text↔image) is different. Retaining "MMDiT" in the name implies fidelity to the MMDiT architecture that is not accurate. The "-k" conventional suffix is also unclear whether it refers to K factors or K streams.
- **Severity**: P1
- **Resolution**: Renamed to "AllToAllFactorStreamDiT" which accurately describes the architecture: all-to-all attention between K factor streams. The previous MMDiT-k name is deprecated. The baseline fidelity plan now reflects this rename.
- **Affected documents**: `docs/research_audit/07_BASELINE_FIDELITY_PLAN.md`, `docs/research_audit/10_IMPLEMENTATION_BACKLOG.md`
- **Validation rule**: `rg "MMDiT-k\|mmdit.k" docs/research_audit/` must return results only in the baseline rename table (deprecation context) or correction registers. `rg "AllToAllFactorStreamDiT" docs/research_audit/` must appear as the active baseline name.
- **Status**: RESOLVED

### CORR-025: Provenance — Added spec_freeze_base_commit, Removed Ambiguous audit_commit

- **Source document**: `docs/research_audit/spec/provenance.yaml`
- **Original text/claim**: The provenance.yaml used `audit_commit` as a general field without distinguishing between the code reference commit, the audit v1 commit, the audit v2 commit, and the spec freeze base commit.
- **Problem**: A single "audit_commit" field is ambiguous — it could refer to the code commit being audited, the audit v1 snapshot, the audit v2 revision, or the base for the spec freeze. Different documents interpreted "audit_commit" differently, leading to cross-references that contradicted each other. For example, the README used "Audit commit: aa14213" while provenance.yaml and manifest could have different commit hashes.
- **Severity**: P1
- **Resolution**: provenance.yaml now includes explicit fields: `code_reference_commit` (c6cc096 — the code being audited), `audit_v1_commit` (aa14213 — first-pass audit), `audit_v2_commit` (c700d34 — second-pass audit with corrections), `spec_freeze_base_commit` (c700d34 — the commit on top of which the v3.0 spec freeze is applied). The ambiguous "audit_commit" field is removed. The `spec_freeze_commit` field remains null until formal freeze approval. The reporting policy is explicit: freeze commit hash reported in agent response, not self-embedded.
- **Affected documents**: `docs/research_audit/spec/provenance.yaml`, `docs/research_audit/audit_manifest.yaml`, `docs/research_audit/README.md`, `docs/research_audit/00_EXECUTIVE_VERDICT.md`
- **Validation rule**: `spec/provenance.yaml` must contain `code_reference_commit`, `audit_v1_commit`, `audit_v2_commit`, `spec_freeze_base_commit`, `spec_freeze_commit`. No field named `audit_commit` without disambiguation.
- **Status**: RESOLVED

---

## Summary

| Metric | Value |
|--------|-------|
| Total corrections | 25 |
| P0 (blocking) | 4 |
| P1 (major) | 13 |
| P2 (minor) | 8 |
| RESOLVED | 25 |
| OPEN | 0 |
| Source documents affected | 18 |
| Validation rules defined | 25 |

All corrections are resolved as of the v3.0 specification freeze. No open corrections remain. Future corrections must be appended to this register with sequential CORR-026+ IDs.
