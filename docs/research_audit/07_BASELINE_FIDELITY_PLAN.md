# 07 — Baseline Fidelity Plan

**Principle**: Baseline names must match actual published methods only when faithfully
implemented. If our implementation deviates from the canonical method, the baseline must
be renamed or brought into compliance. This ensures experimental integrity and prevents
spurious comparisons.

---

## 1. Current Baseline Audit

### SDiT

| Field | Detail |
|---|---|
| **Original paper** | Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023) |
| **Key mechanism** | adaLN-Zero: adaptive layer norm with zero-initialized final linear projection. Each transformer block uses adaLN modulated by timestep + class embedding. The "Zero" initialization ensures identity at t=0. |
| **Current code** | `DiTBlock` without adaLN-Zero. Uses standard adaLN (scale + shift), missing the zero-initialized final projection that is a critical component of the canonical DiT design. |
| **Match / Mismatch** | **MISMATCH** — Missing adaLN-Zero. The zero-init is not a minor detail; it affects training dynamics and final performance. Without it, the model is not a valid DiT baseline. |
| **Decision** | **IMPLEMENT adaLN-Zero** → rename to `DiT` (canonical). Recommended over renaming to `SimpleAdaLNDiT` because: (a) adaLN-Zero is a small implementation change with large fidelity impact, (b) canonical DiT is the standard transformer-based diffusion baseline, (c) renaming invites reviewer questions about why adaLN-Zero was omitted. |
| **Action** | Add zero-initialized final linear projection in each `DiTBlock`. Initialize output projection weights to zero. Verify that at t=0, the block output equals identity. |

### EncDiff

| Field | Detail |
|---|---|
| **Original paper** | "EncDiff: A Disentangled Diffusion Model via Concept Tokens" (NeurIPS 2024) |
| **Key mechanism** | Concept tokens encode factor-specific semantics. Cross-attention between image patch tokens and concept tokens injects factor information. Factor-specific concept tokens are learned. |
| **Current code** | Cross-attention conditioning from factor embeddings to patch tokens, but no concept token mechanism (no learned per-factor token embeddings, no token selection/activation per factor). |
| **Match / Mismatch** | **MISMATCH** — Missing concept tokens. Our implementation is generic cross-attention conditioning, not EncDiff. |
| **Decision** | **RENAME to `CrossAttnDiT`**. Optionally: implement actual EncDiff as an extra baseline in Stage 4 for direct comparison. |
| **Action** | Rename model class. If Stage 4 resources permit, implement EncDiff with: (a) learned concept token embeddings per factor, (b) token activation gated by conditioning set, (c) cross-attention from activated concept tokens to patch tokens. |

### MMDiT-k

| Field | Detail |
|---|---|
| **Original paper** | Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Stable Diffusion 3, ICML 2024) |
| **Key mechanism** | MMDiT (Multi-Modal Diffusion Transformer): separate text and image token streams with joint attention. Text tokens and image tokens attend to each other in designated attention layers. |
| **Current code** | Multi-stream transformer processing K factor streams with cross-stream attention. Structurally similar to MMDiT's multi-stream design but applied to factor conditioning rather than text-image. |
| **Match / Mismatch** | **REASONABLE APPROXIMATION** — Architecture follows MMDiT-style multi-stream design but is applied to factor streams (K streams) rather than text + image (2 streams). |
| **Decision** | **KEEP NAME** but note in text: "MMDiT-style architecture" or "MMDiT-k," not "literal MMDiT." The suffix `-k` signals the K-stream generalization. Add a footnote: "We adopt the multi-modal joint-attention transformer block design from MMDiT, generalized from 2 streams (text, image) to K factor streams." |
| **Action** | Add architecture description clarifying the generalization. No code changes needed. |

### CoInD

| Field | Detail |
|---|---|
| **Original paper** | "CoInD: Conditional Independence for Diffusion Models" (OpenReview, 2025) |
| **Key mechanism** | Fisher divergence loss that penalizes conditional dependence between factor-specific score fields. The loss encourages ∂ε(x_t, i)/∂c_j ≈ 0 for i ≠ j, enforcing that factor i's score field does not depend on factor j. |
| **Current code** | Independent per-factor denoiser streams, but **no Fisher divergence loss**. The architecture is structurally similar (separate streams) but the key contribution of CoInD — the conditional independence objective — is absent. |
| **Match / Mismatch** | **MISMATCH** — Missing Fisher divergence loss. The architecture is similar but the defining mechanism is absent. Calling this CoInD without the Fisher loss is misleading. |
| **Decision** | **RENAME to `IndependentStreamDiT`**. Optionally implement actual CoInD as extra baseline in Stage 4: add Fisher divergence as auxiliary loss to IndependentStreamDiT. |
| **Action** | Rename model class. Add Fisher divergence implementation plan to Stage 4 scope: (a) compute ∂ε(x_t, i)/∂c_j via autograd, (b) penalize squared gradient norm for i ≠ j, (c) balance with denoising loss via λ_Fisher hyperparameter. |

### CF-DiT

| Field | Detail |
|---|---|
| **Original paper** | Ho & Salimans, "Classifier-Free Diffusion Guidance" (NeurIPS 2021 workshop) |
| **Key mechanism** | Joint training with conditional and unconditional (null) inputs. At inference: ε̂ = ε_uncond + w·(ε_cond − ε_uncond). Null token is a learned embedding. |
| **Current code** | Null token index = 0 for each factor embedding. Collision: class 0 is a valid factor value, so null token shares embedding slot with a real class. Factor dropout is all-or-nothing (all factors dropped together or all kept). |
| **Match / Mismatch** | **PARTIAL MATCH with BUGS**: (1) Null token collides with class 0. (2) Factor-wise independent dropout missing. |
| **Decision** | **FIX BOTH**. CF-DiT name is appropriate (the mechanism is CFG, correctly applied at factor level), but the current implementation has correctness issues. No rename needed; fix the bugs. |
| **Action** | (1) Add dedicated null index per factor embedding: `nn.Embedding(size+1, dim)` where index 0 = null, indices 1..size = valid factor values. (2) Implement factor-wise independent dropout: each factor is independently dropped with probability p_drop during training. At inference, each factor independently uses ε_cond_k or ε_uncond_k. |

---

## 2. Summary Table

| Current Name | Match Status | Decision | New Name (if renamed) |
|---|---|---|---|
| SDiT | MISMATCH (missing adaLN-Zero) | IMPLEMENT adaLN-Zero | DiT (canonical) |
| EncDiff | MISMATCH (missing concept tokens) | RENAME | CrossAttnDiT |
| MMDiT-k | REASONABLE (MMDiT-style, not literal) | KEEP NAME + add note | MMDiT-k (unchanged) |
| CoInD | MISMATCH (missing Fisher divergence loss) | RENAME | IndependentStreamDiT |
| CF-DiT | PARTIAL (null collision + no per-factor dropout) | FIX BUGS | CF-DiT (unchanged) |

---

## 3. Additional Baselines to Consider

These are not currently implemented. Priority for Stage 3–4 scope:

| Baseline | Rationale | Priority |
|---|---|---|
| **Canonical DiT** (with adaLN-Zero) | Standard transformer diffusion baseline. Required for fair comparison. | **P0** |
| **Actual CoInD adaptation** | Implement Fisher divergence loss on top of IndependentStreamDiT. Tests whether loss-based CI is sufficient vs. architectural gating. | **P1** |
| **DisDiff adaptation** | Implement factor score field decomposition per DisDiff (NeurIPS 2023). Closest architectural competitor; critical for novelty claim defense. | **P1** |
| **Closest concept bottleneck diffusion** | Implement CBDiffuse-style concept bottleneck with concept-level conditioning. Tests concept-level vs. path-level factor gating. | **P2** |
| **Simple AdaLN DiT** (no zero-init) | Ablation baseline to measure the contribution of adaLN-Zero alone. Useful for component analysis. | **P2** |

---

## 4. Fairness Regime

All baselines must be evaluated under a three-way fairness protocol:

### 4.1. Parameter-Matched Configuration

- Match total parameter count (within ±5%)
- Useful for architecture-level comparison (isolation comes from structure, not scale)
- Favor FGR if FGR achieves better isolation at same parameter budget

### 4.2. FLOP-Matched Configuration

- Match FLOPs per denoising step (within ±5%)
- Useful for compute-efficiency comparison
- Likely favors shared-trunk architectures (DiT, CF-DiT, CrossAttnDiT) over independent experts
- FGR with Architecture B should be competitive here

### 4.3. Best-Practice Configuration

- Each baseline configured with its optimal hyperparameters (no parameter/FLOP constraint)
- Represents "best possible" performance for each method
- Useful for ceiling comparison: can FGR achieve better factor isolation than any existing method at any scale?

### 4.4. Reporting

Each result table must report:
- Parameter count (#M), FLOPs/step, VRAM peak (GB)
- Denoising quality metrics: FID, sFID, IS, Precision, Recall
- Factor isolation metrics: K×K leakage matrix, Intervention Success Rate (ISR), Conditional Independence Violation (CIV)
- Configuration regime (Param-matched / FLOP-matched / Best-practice)
