# 08 — Data and Evaluation Protocol

## Dataset Roles and Rationale

| Dataset | Resolution | Factors | Factor Values | Role | Why |
|---------|-----------|---------|---------------|------|-----|
| dSprites | 64×64 grayscale | shape(3), scale(6), rotation(40), pos_x(32), pos_y(32) | 3×6×40×32×32 = 737,280 | Code sanity + simple edit | Small, fast, well-studied. Factor independence means no graph structure to confound. |
| 3DShapes | 64×64×3 RGB | floor_hue(10), wall_hue(10), object_hue(10), scale(8), shape(4), orientation(15) | 10×10×10×8×4×15 = 480,000 | Multi-factor RGB | Realistic colors, 6 factors, factor correlations exist (floor_hue ~ wall_hue in dataset generation). Tests whether FGR handles correlated factors. |
| Causal3DIdent | 224×224×3 RGB | 7 continuous factors with known SCM DAG | Continuous (sampled from SCM) | Known-SCM graph correctness | Provides ground-truth causal graph. Essential for testing whether correct DAG > wrong DAG. Contains both independent and causally-dependent factors. |

## Factor Assignment Policy

### dSprites

| Factori | Role | Rationale |
|----------|------|-----------|
| shape (3) | Factor stream 0 | Semantic property of interest |
| scale (6) | Factor stream 1 | Semantic property of interest |
| rotation (40) | Factor stream 2 | Semantic property of interest |
| pos_x (32) | **Base stream (nuisance)** | Position variation is nuisance. Do NOT treat as factor stream initially. |
| pos_y (32) | **Base stream (nuisance)** | Same as pos_x. |

Position x/y are suppressed into the base stream. This tests whether the base stream correctly absorbs nuisance while factor streams specialize to shape/scale/rotation. If the base stream fails (position leaks into factor streams), add position as factor streams 3/4 as a diagnostic follow-up experiment.

### 3DShapes

All 6 factors (floor_hue, wall_hue, object_hue, scale, shape, orientation) become factor streams 0-5. No base stream assignment by default — the base stream only handles x_t noise prediction in standard configuration. The graph structure (see below) encodes plausibly-correct dependencies.

### Causal3DIdent

All 7 factors become factor streams 0-6. The ground-truth SCM defines the edge set. No base stream factor assignment — base stream is nuisance-only.

## Splits

### S0: IID Random Split

Standard 80/10/10 train/val/test, uniform random sampling from the full dataset Cartesian product. Baseline split for all experiments. Used for model development and sanity checks.

### S1: Held-Out Pair Combinations

Partition factor-value pair combinations into train and test. For K=3 factors (dSprites shape/scale/rotation): train on all (shape_i, scale_j) pairs, test on held-out (shape_i, scale_j) pairs that were excluded.

Construction:
```python
all_shape_scale = [(i,j) for i in range(3) for j in range(6)]  # 18 pairs
train_pairs = random.sample(all_shape_scale, 14)                # ~78%
test_pairs = [p for p in all_shape_scale if p not in train_pairs]  # ~22%
```
Third factor rotation: use all values in both train and test.

### S2: Held-Out Triple Combinations

Extension of S1 to triplets. For K=3:
```python
all_triplets = [(i,j,k) for i in range(3) for j in range(6) for k in range(40)]  # 720
train_triplets = random.sample(all_triplets, 576)  # 80%
test_triplets = [p for p in all_triplets if p not in train_triplets]  # 20%
```

### S3: Systematic Compositional Split

Hold out entire factor value(s) from training. For dSprites: train on shapes [0,1], test on shape [2]. Tests whether factor path specializes to the factor concept rather than memorizing individual values.

### Split Implementation Constraints

All split indices are deterministic — seeded RNG at split construction time. Indices stored as JSON manifest alongside dataset. Splits are dataset-specific, not model-specific. Same split indices used across all models for fair comparison.

## Lazy Loading Protocol

### HDF5 Access Pattern

```python
class LazyHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, split_indices):
        self.h5_path = h5_path
        self.indices = split_indices

    def __getitem__(self, idx):
        # Open HDF5 handle PER WORKER (pid-based caching)
        h5 = self._get_worker_handle()
        real_idx = self.indices[idx]
        image = h5['images'][real_idx]       # memory-mapped slice
        factors = h5['factors'][real_idx]    # memory-mapped slice
        return torch.from_numpy(image), torch.from_numpy(factors)

    def _get_worker_handle(self):
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = h5py.File(self.h5_path, 'r', swmr=True)
        return self._handles[pid]
```

Key properties:
- Per-worker HDF5 file handles (not shared across workers, avoids locking)
- Memory-mapped access (no pre-load into RAM)
- Deterministic split indices (same across runs, seeded)
- No image preprocessing in dataset class (handled in model forward or transform pipeline)
- `swmr=True` for safe concurrent reads

### 3DShapes-Specific Notes

3DShapes provides `.h5` with `images` (480,000, 64, 64, 3) and factor labels. Memory-mapped access is essential — full dataset is ~2.4 GB raw.

### Causal3DIdent-Specific Notes

Causal3DIdent images are stored individually or in batched HDF5. Factor values are sampled from the SCM. The SCM graph is loaded alongside the dataset and passed to `FGRDiT` at construction. Factor values are continuous — no discretization for conditioning, though discrete binning may be used for evaluation metrics.

## Oracle Pipeline

### Architecture

Simple ResNet-18 or ConvNeXt-Tiny classifier conditioned on the relevant factors. One classifier per dataset. Multi-head output: K heads each predicting factor_k.

### Training Protocol

```
for epoch in range(max_epochs):
    for x, f in train_loader:
        f_hat = oracle(x)
        loss = Σ_k CrossEntropy(f_hat_k, f_k)
        loss.backward()
        optimizer.step()

    # Validation after each epoch
    val_acc_k = accuracy(f_hat_val, f_val) per factor
    if best_val_acc: save_checkpoint()
    if patience exceeded: early_stop()
```

### Required Outputs

| Artifact | Description |
|----------|-------------|
| `oracle_checkpoint.pt` | Full model state + config |
| `oracle_confusion.npz` | K factor-level confusion matrices |
| `oracle_calibration.json` | ECE per factor, reliability diagrams |
| `oracle_version.txt` | Commit hash, training date, dataset hash |

### Generated-Image Robustness Check

Run oracle on FGR-generated images where the conditioning factor values are known. Compare oracle accuracy on generated images vs. ground-truth dataset images. A significant drop (>5%) indicates domain shift or generation artifacts that the oracle misinterprets.

### Oracle Versioning

Oracle checkpoint hash stored in evaluation output. If oracle retrained, old eval results flagged as incomparable. Oracle is part of the evaluation pipeline, not a replaceable component.

## Paired Evaluation via NoiseTrace

### Principle

Common-random-number coupling: comparing two generation runs uses the same underlying randomness to isolate the effect of the intervention.

```
x_original = sample(checkpoint, original_factors, trace=trace_A)
x_edited   = sample(checkpoint, edited_factors,   trace=trace_A)  # same trace!
```

### NoiseTrace Representation A: Deterministic DDIM (η=0)

```python
@dataclass
class NoiseTrace:
    x_T: torch.Tensor          # [B, C, H, W] initial noise
    sampler: str = "ddim"
    eta: float = 0.0
    steps: int = 50
    # No per-step noise stored — DDIM η=0 is fully deterministic given x_T.
```

**Recommended for evaluation.** No per-step randomness to couple. Store only x_T per sample. Counter-based RNG (Philox) used for x_T generation to enable exact reconstruction.

### NoiseTrace Representation B: Stochastic DDPM with Shared Trace

```python
@dataclass
class NoiseTraceStochastic:
    x_T: torch.Tensor               # [B, C, H, W] initial noise
    sampler: str = "ddpm"
    steps: int = 1000
    # Counter seeds per (sample, step) instead of full noise tensors
    seed_grid: list[int]            # [B * steps] counter seeds
    rng_algorithm: str = "philox"
```

On-the-fly deterministic noise generation:
```python
def generate_step_noise(seed, shape, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device)
```

Memory: 4 bytes per (sample, step) vs 4×C×H×W bytes per (sample, step). For 1000 samples × 1000 steps × 64×64×3: seeds = 4MB vs noise = 12GB.

### Invariant: Same NoiseTrace → Same Output

**Requirement**: `sample(model, condition, trace=A)` and `sample(model, condition, trace=A)` must produce identical tensors down to float32 precision (bit-exact). This is a Phase Gate 0 requirement. Violation indicates non-deterministic ops (atomicAdd, cudnn benchmarking, etc.).

## Evaluation Protocols

### Protocol 1: Factor Edit (factor_edit)

```
Goal: measure target efficacy and off-target leakage when changing one factor value

Input:  x_T (from NoiseTrace), original factors f (K values)
Output: generated image x_0

For each factor i ∈ [0..K-1]:
    For each pair (v_old, v_new) where v_old ≠ v_new:
        x_orig = sample(model, f, trace)
        f' = f.copy(); f'[i] = v_new
        x_edit = sample(model, f', trace)   # SAME trace

        target_success[i] = 1 if Oracle_i(x_edit) == v_new else 0
        for j ≠ i:
            leakage[i,j] = 1 if Oracle_j(x_edit) ≠ Oracle_j(x_orig) else 0
        noop_stability = Oracle_i(x_orig) == v_old  # oracle sanity check

Aggregate:
    target_accuracy[i] = mean(target_success[i])
    leakage_matrix[i,j] = mean(leakage[i,j])
    off_target_mean[i] = mean(leakage[i,j] for j≠i)
```

**Constraint**: v_new ≠ v_old ALWAYS enforced. Use offset sampling: v_new = (v_old + uniform(1, N_i-1)) % N_i. Never use independent randint (collision possible).

### Protocol 2: Full Source Cut Invariance (path_ablation)

```
Goal: verify that cutting factor path i removes all direct influence of factor i

For each factor i:
    spec_cut = InterventionSpec(
        mode="path_ablation",
        output_gates=[..., 0 at position i, ...]
    )
    x_ablate = sample(model, f, trace, intervention=spec_cut)

    For all v_new ≠ f[i]:
        f' = f.copy(); f'[i] = v_new
        spec_cut_2 = InterventionSpec(
            mode="path_ablation",
            output_gates=[..., 0 at i, ...]
        )
        x_ablate_alt = sample(model, f', trace, intervention=spec_cut_2)

        assert torch.allclose(x_ablate, x_ablate_alt, atol=1e-5)
        # Path Non-Interference Theorem: output must be independent of f[i] when cut

    # Base invariance check: the cut output should differ from uncut when factor differs
    x_uncut = sample(model, f, trace)  # no intervention
    assert not torch.allclose(x_ablate, x_uncut, atol=1e-2)
    # Cutting a path should change the output (unless the path was already dead)
```

### Protocol 3: Neural Graph Surgery (graph_surgery)

```
Goal: test whether incoming-edge cut + factor edit respects graph semantics

For DAG G = (V, E) where V = {0..K-1}:
    For each factor i:
        spec_gs = InterventionSpec(
            mode="graph_surgery",
            edited_factors={i: v_new},
            incoming_cut={i}       # cut all edges INTO i
        )
        x_surgery = sample(model, f, trace, intervention=spec_gs)

        # Direct contribution: should reflect v_new (no incoming interference)
        target_accuracy[i] = P[Oracle_i(x_surgery) == v_new]

        # Children of i: outgoing edges preserved, so children SHOULD change
        for j in Ch(i):
            x_surgery_j_diff = (Oracle_j(x_surgery) != Oracle_j(x_orig))

        # Non-descendants: should be invariant
        non_desc = all nodes except i and descendants of i
        for j in non_desc:
            leakage_surgery[i,j] = P[Oracle_j(x_surgery) != Oracle_j(x_orig)]

Compare: graph_surgery leakage vs factor_edit leakage.
Prediction: graph_surgery has LOWER leakage on non-descendants.
```

## Metrics

### M1: Target Desired-Value Success

```
S_i(v→v') = P[Oracle_i(x_edit) = v']
```
Probability that the edited image is classified as the desired target value. Aggregate over all v→v' pairs to get per-factor target accuracy.

### M2: Off-Target Change (Leakage Matrix)

```
L_ij = P[Oracle_j(x_edit_i) ≠ Oracle_j(x_original)]
```
For i=j: this measures how often changing factor i changes factor i's oracle prediction (should be high for successful edits, but NOT 1.0 — generation stochasticity may affect oracle).
For i≠j: leakage — how often changing factor i inadvertently changes factor j.

Reported as K×K matrix with diagonal highlighted.

### M3: No-Op Change

```
N_i = P[Oracle_i(x_same) ≠ Oracle_i(x_original)]
```
where x_same is generated with same condition + same NoiseTrace. Should be exactly 0 (same NoiseTrace → same output). If >0, oracle is noisy or generation has non-determinism. Used as baseline noise floor.

### M4: Generation Quality

Two variants:
- **Oracle conditional accuracy**: P[∀i: Oracle_i(x_gen) = f_i] for conditioned generation. Measures whether the generated image faithfully represents the requested factors.
- **FID**: Fréchet Inception Distance vs. ground-truth dataset, if a suitable reference distribution exists (dSprites and 3DShapes: yes; Causal3DIdent: limited reference samples). Use Clean-FID implementation.

### M5: Minimality (LPIPS)

```
LPIPS(x_orig, x_edit) — should be small for targeted edits, large for holistic changes
```
Perceptual distance between original and edited image. Small LPIPS + high target success = precise edit. Large LPIPS + high target success = holistic change (acceptable). Large LPIPS + low target success = bad edit.

## Statistical Plan

### Training Replicates

| Experiment Phase | Seeds | Rationale |
|-----------------|-------|-----------|
| Development | 1-2 | Fast iteration |
| Pilot (Stage 4-5) | 2-3 | Initial signal |
| Confirmatory (Stage 6) | 5 | Main results, CIs |

### Evaluation Sample Size

1,000 evaluation samples per seed × condition. For factor_edit with K=3 and 3 value pairs: 3 × 3 × 1000 = 9,000 generated images per seed.

### Confidence Intervals

- **Paired bootstrap 95% CI** for continuous metrics (LPIPS, FID): resample evaluation samples with replacement (B=10,000), compute metric per bootstrap replicate, report 2.5th and 97.5th percentiles. Paired: resample indices, not independently per condition.
- **Binomial (Clopper-Pearson) 95% CI** for accuracy metrics (target success, leakage proportions).
- **Effect size**: Cohen's h for proportions, Cohen's d for continuous. Report alongside CIs.

### Multiple Comparison Correction

When comparing M models × K factors × S splits: apply Benjamini-Hochberg FDR correction (α=0.05) across the family of factor-wise leakage comparisons. Report both raw and corrected p-values.

### Reporting Standard

Per-metric format:
```
Metric name: mean [95% CI lower, 95% CI upper]
paired bootstrap, B=10,000, 5 seeds, 1,000 samples/seed
```

## Negative Controls

### NC1: Same Condition + Same NoiseTrace → Identical Output

```
N_samples paired: (x_T, condition) → generate twice
Metric: max |output_1 - output_2|
Threshold: < 1e-6 (bit-exact)
Failure: non-deterministic ops in pipeline
```

### NC2: Same Condition + Different NoiseTrace → Randomness Baseline

```
x_1 = sample(model, f, trace_A)
x_2 = sample(model, f, trace_B)
Metric: mean pairwise LPIPS, mean oracle disagreement
Expected: > 0 (establishes stochasticity ceiling)
```

### NC3: Shuffled Factor Labels

```
f_shuffled = permute(f)  # f_shape → f_scale, f_scale → f_rotation, f_rotation → f_shape
x_shuffled = sample(model, f_shuffled, trace)
Metric: factor oracle accuracy (should be near chance)
Expected: factor paths learn alignment to correct factor input
Failure: model ignores factor embedding → path specialization not meaningful
```

### NC4: Graph Misspecification Controls

| Graph | Edges | Expected Result |
|-------|-------|-----------------|
| Correct | Ground-truth edges (Causal3DIdent SCM) | Lowest non-descendant leakage |
| Empty | No edges (all factors independent) | Higher leakage (missing parent→child pathways) |
| Complete | All (j,i) for j≠i | Highest leakage or worst (spurious edges cause interference) |
| Reversed | All (i,j) where (j,i) in correct graph | Higher leakage than correct, potentially worse than empty |
| Random | Random DAG with same edge density | Intermediate leakage |

### NC5: Fake Intervention (Must-Enforce)

```
FAKE: v_new == v_old (same value)
MUST FAIL: assertion error at InterventionSpec construction
"If editing nothing, report an error" — prevents silent no-ops in eval loop.
```

## Terminology Policy

| Term | Use Case |
|------|----------|
| paired-noise evaluation | Standard term for shared NoiseTrace comparison |
| counterfactual | Allowed ONLY in limited sense: "same exogenous randomness, different factor condition" |
| causal / do-operator | NOT used unless SCM equivalence proven (L4 claim only) |
| factor edit | Preferred over "counterfactual edit" |
| path cut / ablation | Preferred over "causal ablation" |
| leakage | Preferred over "causal effect" |
| graph surgery | Preferred over "do-intervention" |

Never use "disentanglement" — FGR does not produce disentangled representations; it produces factor-specific computational paths with verifiable non-interference. These are distinct concepts.
