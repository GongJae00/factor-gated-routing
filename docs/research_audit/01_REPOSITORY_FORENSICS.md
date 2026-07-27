# 01 — Repository Forensics (v2)

**Audit commit**: aa14213 | **Code reference**: c6cc096
**Date**: 2026-07-27 | **Method**: Static analysis + `git ls-files` at c6cc096

## File Inventory (all tracked files at c6cc096)

| File | LOC | Purpose | Risk | Evidence |
|------|-----|---------|------|----------|
| README.md | 207 | Project docs | MEDIUM — `cd gauge-sensitive-inverse-generation` at L49 | VERIFIED |
| MATH_NOTES.md | 70 | 4 propositions | HIGH — Prop 4 false, Prop 1 misleading | VERIFIED |
| .env.example | 13 | Env template | LOW | VERIFIED |
| .gitignore | 44 | Git rules | MEDIUM — `h5` is too narrow (matches file `h5`, not `*.h5`); `checkpoint*` is oddly placed | VERIFIED |
| pyproject.toml | 31 | Package metadata | LOW — no PyYAML dependency for YAML config plan | VERIFIED |
| uv.lock | — | Dependency lock | LOW | VERIFIED |
| src/__init__.py | 1 | Pkg init | LOW | VERIFIED |
| src/model.py | 135 | FGRDiT, FGRStream | **HIGH** — P0 defects listed below | VERIFIED |
| src/baselines.py | 252 | 5 baselines | **HIGH** — naming: CoInD/EncDiff not faithful | VERIFIED |
| src/train.py | 226 | Training loop | **HIGH** — YAML unused, gate=1 always, import bug | VERIFIED |
| src/evaluate.py | 147 | Evaluation | **HIGH** — unpaired sampling, overwrite bug | VERIFIED |
| src/diffusion.py | 56 | DDPM sampler | HIGH — unpaired, DDPM formula unverified vs Ho et al. 2020 | VERIFIED |
| src/oracle.py | 22 | OracleClassifier | MEDIUM — no training pipeline | VERIFIED |
| src/config.py | 69 | Config dataclasses | **HIGH** — import-time env enforcement at L31 | VERIFIED |
| src/dataset.py | 69 | DSprites, 3DShapes | HIGH — full RAM, position x/y missing | VERIFIED |
| src/utils.py | 133 | DiTBlock, CrossAttnBlock | MEDIUM — adaLN-Zero incomplete (no residual gate α) | VERIFIED |
| src/registry.py | 11 | MODEL_REGISTRY | LOW | VERIFIED |
| configs/dsprites.yaml | 14 | dSprites config | CRITICAL — NEVER USED by train.py | VERIFIED |
| configs/shapes3d.yaml | 14 | 3DShapes config | CRITICAL — NEVER USED by train.py | VERIFIED |
| scripts/run_fgr.sh | 21 | Single model | LOW | VERIFIED |
| scripts/run_full_experiment.sh | 33 | Sequential train | LOW | VERIFIED |
| tests/test_fgr_model.py | 51 | Model smoke | LOW — no semantic tests | VERIFIED |
| tests/test_fgr_baselines.py | 77 | Baseline smoke | LOW | VERIFIED |
| tests/test_fgr_diffusion.py | 45 | Diffusion+oracle | LOW | VERIFIED |
| tests/test_fgr_dataset.py | 42 | Dataset tests | LOW — skips without data | VERIFIED |

## Adjudicated Hypotheses (H-001 through H-085)

### Code-activated hypotheses (H-001 through H-012)

| ID | Hypothesis | Status | Confidence | Evidence |
|----|-----------|--------|------------|----------|
| H-001 | dSprites config: use_cross_attn=false | **VERIFIED** | 1.00 | configs/dsprites.yaml:L12 |
| H-002 | 3DShapes config: use_cross_attn=false | **VERIFIED** | 1.00 | configs/shapes3d.yaml:L12 |
| H-003 | train.py reads hardcoded dicts, not YAML | **VERIFIED** | 1.00 | src/train.py:L18-28 DSPRITES_CFG/SHAPES3D_CFG dicts |
| H-004 | Default FGR = independent streams + output gating | **VERIFIED** | 1.00 | `use_cross_attn=False`, `dag_edges=[]` |
| H-005 | full_ca is lower-triangular, not all-to-all | **VERIFIED** | 0.95 | src/model.py:L116-117 — `if ca_mode == "full" and i > 0` only sees earlier streams |
| H-006 | No topological sort; parent-behind-child = silent drop | **VERIFIED** | 0.95 | src/model.py:L121 — `if pi < len(stream_outputs)` silently filters |
| H-007 | No DAG validation (cycle/invalid/duplicate) | **VERIFIED** | 1.00 | No validation code exists anywhere |
| H-008 | Final-depth state repeated for all blocks, not layerwise | **VERIFIED** | 1.00 | src/model.py:L124 — `stream(tokens, t_emb, ...)` passes all blocks in one call |
| H-009 | CrossAttnBlock: sequential parent processing, order-sensitive | **VERIFIED** | 0.95 | src/utils.py:L123-128 — `for p_state in parent_states` uses same CA params |
| H-010 | set_inter_stream_ca: new blocks = random init, no weight transfer | **VERIFIED** | 1.00 | src/model.py:L37-38 — `new_block = DiTBlock(...).to(device)` — fresh random |
| H-011 | set_inter_stream_ca: internal state inconsistent after call | **VERIFIED** | 0.90 | src/model.py:L34 — uses `self.use_cross_attn` for branching but doesn't update it |
| H-012 | CA/DAG checkpoint → evaluate.py RuntimeError | **VERIFIED** | 0.95 | src/evaluate.py:L57-62 — strict=False with RuntimeError on unexpected keys |

### Gate and intervention hypotheses (H-013 through H-022)

| H-013 | Graph not stored in checkpoint | **VERIFIED** | 1.00 | src/train.py:L203 — only `model.state_dict()` saved |
| H-014 | factor_names = factor_sizes bug | **VERIFIED** | 0.95 | src/model.py:L79 — `self.factor_names = config.factor_sizes` |
| H-015 | Training-time gate always 1 | **VERIFIED** | 1.00 | src/train.py:L167 — `gates = [1.0] * config.n_factors` |
| H-016 | gate=0 or 0.5 is OOD at inference | **VERIFIED** | 1.00 | Never seen during training |
| H-017 | Gate linear semantics unproven | **VERIFIED** | 1.00 | Shared LN distorts scaling |
| H-018 | Shared LN breaks additivity | **VERIFIED** | 0.95 | `W·LN(Σ g_i h_i)` ≠ `Σ W·LN(g_i h_i)` |
| H-019 | Gate = path ablation, not do-operator | **VERIFIED** | 1.00 | src/model.py:L30 — `x = x * gate` → computational node ablation |
| H-020 | Gate semantics: doc says direct output only, code says child routing too | **VERIFIED** | 0.90 | src/model.py:L124 uses gated `so` in `stream_outputs` → child gets gated state |
| H-021 | Factor edit, ablation, surgery mixed in evaluation | **VERIFIED** | 1.00 | src/evaluate.py:L81-98 — changes label AND gate simultaneously |
| H-022 | Output contribution vs outgoing message vs incoming edge conflated | **VERIFIED** | 1.00 | Single `gates` list controls all via `x * gate` at stream output |

### Math/theory hypotheses (H-023 through H-033)

| H-023 | Prop 1: restricted condition validity | **VERIFIED** | 0.90 | No cross-attn bypass → correct; with bypass → gradient not zero |
| H-024 | Prop 1 child-stream description = code mismatch | **VERIFIED** | 0.85 | Doc says g_i*h_i zeroed but child still sees h_i; code gates first, then passes to child |
| H-025 | Prop 2: no assumptions or proof | **VERIFIED** | 1.00 | No function class, graph, capacity assumptions |
| H-026 | Prop 3: supergraph auto-ignore is generally false | **VERIFIED** | 0.95 | Without sparsity penalty, extra edges can be used |
| H-027 | Prop 4: Lipschitz → monotonicity is false | **VERIFIED** | 1.00 | Rotation ODE counterexample |
| H-028 | Correct bound: Lipschitz sensitivity, not monotonicity | **VERIFIED** | 0.80 | Grönwall inequality approach is correct direction |
| H-029 | Shared x_t = semantic leakage pathway | **VERIFIED** | 1.00 | All streams receive same x_t via patch_embed |
| H-030 | Branch decomposition non-identifiable via reconstruction loss | **VERIFIED** | 0.90 | Gauge freedom: any δ_i with Σ δ_i=0 preserves output |
| H-031 | Factor-specific embedding alone ≠ branch identifiability | **VERIFIED** | 0.85 | Weak identifiability from input restriction |
| H-032 | Shared LN + head complicates factor contribution interpretation | **VERIFIED** | 0.90 | Nonlinear LN over sum breaks additivity |
| H-033 | Denoiser-level invariance ≠ image-level factor absence | **VERIFIED** | 0.95 | Shared x_t allows other streams to infer factor i |

### Sampling/evaluation hypotheses (H-034 through H-045)

| H-034 | Normal vs intervention: different x_T | **VERIFIED** | 1.00 | src/evaluate.py:L83-85, L94-98 — two independent `sample_images` calls |
| H-035 | Timestep noise also different | **VERIFIED** | 1.00 | src/diffusion.py:L54 — `torch.randn_like(x)` per step |
| H-036 | Pixel diff reflects sampling noise, not intervention | **VERIFIED** | 1.00 | Consequence of H-034, H-035 |
| H-037 | Gate sweep uses independent noise per gate value | **VERIFIED** | 1.00 | src/evaluate.py:L124-128 — new `sample_images` call per gate |
| H-038 | new_val may equal old_val | **VERIFIED** | 1.00 | src/evaluate.py:L88-89 — `torch.randint(0, S_i, ...)` |
| H-039 | nonintervention_stability = pixel threshold, not factor-level | **VERIFIED** | 0.95 | src/evaluate.py:L102-104 — `change_mag < 0.05` |
| H-040 | cond_accuracy overwritten per loop | **VERIFIED** | 0.90 | src/evaluate.py:L109-114 — recomputed each intervene_idx loop |
| H-041 | Normal generation repeated per intervention factor | **VERIFIED** | 1.00 | src/evaluate.py:L83-85 inside `for intervene_idx` loop |
| H-042 | DDPM formula unverified vs Ho et al. 2020 | **UNVERIFIED** | 0.50 | Formula appears correct but no explicit verification test |
| H-043 | n_steps not dividing 1000: off-by-one possible | **VERIFIED** | 0.80 | src/diffusion.py:L31 — `dt = T_1000 // n_steps` uses floor division |
| H-044 | Same condition + same randomness identity test missing | **VERIFIED** | 1.00 | No such test exists in tests/ |
| H-045 | No paired statistics | **VERIFIED** | 1.00 | All comparisons use independent samples |

### Baseline fidelity hypotheses (H-046 through H-054)

| H-046 | CoInDDiT ≠ published CoInD | **VERIFIED** | 1.00 | No Fisher divergence loss anywhere |
| H-047 | EncDiffDiT ≠ published EncDiff | **VERIFIED** | 0.98 | No concept tokens, single repeated vector cross-attention |
| H-048 | EncDiff repeated conditioning = degenerate attention | **VERIFIED** | 0.95 | src/baselines.py:L64 — identical K,V → uniform weights → bias injection |
| H-049 | CF-DiT zero = actual valid class | **VERIFIED** | 1.00 | src/baselines.py:L236 — `torch.zeros_like(factor_classes)` |
| H-050 | All-or-nothing dropout, not factor-wise | **VERIFIED** | 1.00 | src/baselines.py:L235 — single `drop` mask for all factors |
| H-051 | SDiT ≠ canonical DiT adaLN-Zero | **VERIFIED** | 0.95 | src/utils.py:L76-99 — no residual gating α, no zero-init |
| H-052 | Params matched but FLOPs/depth differ | **VERIFIED** | 0.85 | FGR: 3×4 blocks parallel, SDiT: 12 sequential |
| H-053 | CA parameters change FGR param count | **VERIFIED** | 0.90 | CrossAttnBlock adds ~3.2M for K=3 |
| H-054 | Using paper names without faithful implementation = integrity issue | **VERIFIED** | 0.95 | Academic integrity concern |

### Dataset/oracle hypotheses (H-055 through H-063)

| H-055 | Random split ≠ compositional generalization | **VERIFIED** | 1.00 | All factor combos exist in both train and test |
| H-056 | Position x/y not conditioned in dSprites | **VERIFIED** | 1.00 | src/dataset.py:L20 — `[:, 1:4]` = shape, scale, orientation only |
| H-057 | Position variation absorbed as nuisance | **UNVERIFIED** | 0.60 | Plausible but no empirical evidence yet |
| H-058 | dSprites/3DShapes factors are independent → DAG contribution weak | **VERIFIED** | 0.90 | Both datasets have independent factor generation |
| H-059 | 3DShapes full RAM load | **VERIFIED** | 0.95 | ~22 GB for 480K×64×64×3 float32 |
| H-060 | dSprites full RAM load | **VERIFIED** | 1.00 | src/dataset.py:L8-9 — `np.load` loads entire file |
| H-061 | No oracle held-out accuracy/confusion matrix | **VERIFIED** | 1.00 | No oracle training script exists |
| H-062 | Generated-image oracle domain shift unverified | **UNVERIFIED** | 0.50 | No generated images exist yet to test |
| H-063 | Single oracle → classifier-dependent metrics | **VERIFIED** | 0.90 | No ensemble or multi-architecture agreement |

### Engineering hypotheses (H-064 through H-075)

| H-064 | Import-time env enforcement for both datasets | **VERIFIED** | 1.00 | src/train.py:L30-33 — DATASET_PATHS dict evaluated at import |
| H-065 | TrainConfig fields unused | **VERIFIED** | 0.95 | src/config.py:L26-40 — TrainConfig never instantiated in train.py |
| H-066 | mixed_precision dtype mismatch | **PARTIALLY_VERIFIED** | 0.80 | Mismatch exists but reason differs from original claim: src/train.py:L165 uses `torch.amp.autocast("cuda")` with default dtype (fp16), while TrainConfig.mixed_precision="bf16" is declared but never applied. The default autocast dtype does not match the configured one. |
| H-067 | Checkpoint inside log_every nesting | **VERIFIED** | 1.00 | src/train.py:L201 — nested inside `if (step+1) % args.log_every == 0` |
| H-068 | Checkpoint lacks optimizer/scheduler/RNG | **VERIFIED** | 1.00 | src/train.py:L204-205 — only model/EMA state_dict |
| H-069 | No resume training | **VERIFIED** | 1.00 | No --resume flag or state restoration |
| H-070 | Smoke tests only, no semantic checks | **VERIFIED** | 1.00 | tests/ only check shapes and param counts |
| H-071 | Dataset tests skip → "all pass" misleading | **VERIFIED** | 0.95 | tests/test_fgr_dataset.py:L6 — `@requires_dsprites` skip marker |
| H-072 | README stale: repo path, config | **VERIFIED** | 1.00 | L49 — `cd gauge-sensitive-inverse-generation` |
| H-073 | .gitignore vs output dir mismatch | **VERIFIED** | 0.95 | .gitignore has `outputs/` (plural), default output is `output/` (singular). Verified mismatch at c6cc096. |
| H-074 | Generic `src` package name | **VERIFIED** | 0.70 | `src` is non-descriptive but functional; `pyproject.toml` uses `factor-gated-routing` as project name |
| H-075 | No environment lock / seed manifest / data hash | **VERIFIED** | 1.00 | None of these exist |

### New defects (H-076 through H-085 from v1 audit)

| ID | Hypothesis | Status | Confidence | Evidence |
|----|-----------|--------|------------|----------|
| H-076 | README L49 instructs `cd gauge-sensitive-inverse-generation` — stale repo name | **VERIFIED** | 1.00 | README.md L49 at c6cc096 |
| H-077 | `build_config` hardcodes `use_gating = (model_name == "FGR")` — baselines can never use gating | **VERIFIED** | 1.00 | src/train.py:L166-168 |
| H-078 | FGRStream stores `factor_idx` but no validation that stream order = factor order | **VERIFIED** | 0.95 | src/model.py FGRStream.__init__ |
| H-079 | `FGRDiT.factor_names` set to `config.factor_sizes` — naming bug | **VERIFIED** | 0.95 | src/model.py:L79 |
| H-080 | evaluate.py uses `torch.randint(0, S_i, ...)` — ~1/S_i chance new_val == old_val | **VERIFIED** | 1.00 | src/evaluate.py:L88-89 |
| H-081 | evaluate.py raises RuntimeError on unexpected keys; full_ca/DAG checkpoints WILL have unexpected CrossAttnBlock keys | **VERIFIED** | 0.95 | src/evaluate.py:L57-62 |
| H-082 | DiTBlock lacks residual gating (adalN-Zero alpha parameter) — confirmed vs Facebook DiT repo | **VERIFIED** | 1.00 | src/utils.py:L76-99 |
| H-083 | `FGRConfig` and `TrainConfig` in config.py are NEVER instantiated by train.py or evaluate.py | **VERIFIED** | 1.00 | Full codebase grep |
| H-084 | `CosineSchedule.get_alpha_bars` returns alpha_bars directly (not cumprod) — correct for cosine schedule | **VERIFIED** | 1.00 | src/diffusion.py:L13-14 |
| H-085 | AGENTS.md L18 references `fgr/` package — should be `src/` | **PARTIALLY_VERIFIED** | 0.70 | AGENTS.md file now exists at HEAD but git show confirms it did NOT exist at c6cc096. The file content (fgr/ path reference) is real at HEAD, but the file itself was not present at the audited commit. |

## Counts

- **VERIFIED**: 80 (71 from H-001–H-075 individual rows + 9 from H-076–H-084)
- **PARTIALLY_VERIFIED**: 2 (H-066 — mixed_precision mismatch reason differs; H-085 — AGENTS.md not present at c6cc096 but exists at HEAD)
- **UNVERIFIED**: 3 (H-042, H-057, H-062)
- **REFUTED**: 0
- **Total adjudicated**: 85 hypotheses
