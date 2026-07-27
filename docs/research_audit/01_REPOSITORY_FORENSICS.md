# 01 — Repository Forensics

**Commit**: c6cc096 @ factor-gated-routing
**Date**: 2026-07-27
**Method**: Static analysis of all tracked files

## File Inventory

| File | LOC | Purpose | Risk |
|------|-----|---------|------|
| README.md | 207 | Project docs, usage, architecture | MEDIUM — contains stale `cd gauge-sensitive-inverse-generation` (line 49) |
| MATH_NOTES.md | 70 | 4 propositions | HIGH — Prop 4 is false, Prop 1 misleading |
| AGENTS.md | 30 | Codex guidance | LOW — references stale `fgr/` package name |
| .env.example | 13 | Environment template | LOW |
| .gitignore | 44 | Git ignore rules | LOW — too broad: `h5`, `checkpoint*` as bare strings |
| pyproject.toml | 31 | Package metadata | LOW |
| uv.lock | 83.6K | Dependency lock | LOW |
| src/__init__.py | 1 | Package init | LOW |
| src/model.py | 135 | FGRDiT, FGRStream | **HIGH** — P0 defects |
| src/baselines.py | 252 | 5 baselines + build_baseline | **HIGH** — naming fidelity |
| src/train.py | 226 | Training loop | **HIGH** — YAML unused, gate always 1, import bug |
| src/evaluate.py | 147 | Evaluation pipeline | **HIGH** — unpaired sampling, loop overwrite |
| src/diffusion.py | 56 | DDPM sampler | HIGH — unpaired, no validation test |
| src/oracle.py | 22 | OracleClassifier | MEDIUM — no training pipeline |
| src/config.py | 69 | ModelConfig, path resolution | **HIGH** — import-time env enforcement |
| src/dataset.py | 69 | DSprites, 3DShapes loaders | HIGH — full RAM load, position missing |
| src/utils.py | 133 | DiTBlock, CrossAttnBlock, AdaLN | MEDIUM — adaLN-Zero incomplete |
| src/registry.py | 11 | MODEL_REGISTRY | LOW |
| configs/dsprites.yaml | 14 | dSprites config | CRITICAL — NEVER USED by train.py |
| configs/shapes3d.yaml | 14 | 3DShapes config | CRITICAL — NEVER USED by train.py |
| scripts/run_fgr.sh | 21 | Single model runner | LOW |
| scripts/run_full_experiment.sh | 33 | Sequential trainer | LOW |
| tests/test_fgr_model.py | 51 | Model smoke tests | LOW — no semantic checks |
| tests/test_fgr_baselines.py | 77 | Baseline smoke tests | LOW |
| tests/test_fgr_diffusion.py | 45 | Diffusion + oracle tests | LOW |
| tests/test_fgr_dataset.py | 42 | Dataset tests | LOW — skips if data absent |

## 10+ New Defects Found

H-076 (NEW): README line 49 instructs `cd gauge-sensitive-inverse-generation` — stale repo name.
H-077 (NEW): `build_config` in train.py:166-168 hardcodes `use_gating = (model_name == "FGR")` — means baselines can NEVER use gating even if appropriate.
H-078 (NEW): FGRStream.__init__ stores `factor_idx` but `forward` uses `factor_class[:, self.factor_idx]` — `factor_idx` is stream's own index, assuming stream order = factor order. No validation.
H-079 (NEW): `FGRDiT.factor_names` set to `config.factor_sizes` (line 79) — naming bug, should be `config.factor_names` if it existed.
H-080 (NEW): evaluate.py:88-89 use `torch.randint(0, S_i, ...)` — ~1/S_i chance new_val == old_val.
H-081 (NEW): evaluate.py:57-62 raises RuntimeError on unexpected keys, but `full_ca` or DAG checkpoints WILL have unexpected CrossAttnBlock keys.
H-082 (NEW): DiTBlock lacks residual gating (adalN-Zero alpha parameter) — confirmed vs Facebook DiT repo.
H-083 (NEW): `FGRConfig` and `TrainConfig` dataclasses in config.py are NEVER instantiated by train.py or evaluate.py.
H-084 (NEW): `CosineSchedule.get_alpha_bars` returns alpha_bars DIRECTLY (not cumprod'd) — line 13-14 of diffusion.py. This is the cosine schedule alpha_bar, not alphas. VERIFIED correct for cosine.
H-085 (NEW): AGENTS.md line 18 references `fgr/` package — should be `src/`.
