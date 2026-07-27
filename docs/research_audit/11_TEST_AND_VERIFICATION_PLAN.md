# 11 — Test and Verification Plan

**Minimum 25 tests required for correctness confidence before GPU experiments.**

## Graph Tests

| T-01 | **Cycle detection**: DAG with cycle `[(0,1),(1,2),(2,0)]` raises ValueError |
| T-02 | **Invalid node**: Edge `[(0, 10)]` with 3 factors raises ValueError |
| T-03 | **Duplicate edge**: Edge `[(0,1),(0,1)]` raises ValueError or deduplication warning |
| T-04 | **Self-loop**: Edge `[(0,0)]` raises ValueError |
| T-05 | **Topological validity**: Correct DAG produces output matching expectation |
| T-06 | **Node permutation**: Renumbering nodes + permuting embeddings preserves output |

## Intervention Tests

| T-07 | **Output gate zero**: `o_i=0` → factor embedding invariance (all gates 0 vs 1) outputs are different |
| T-08 | **Incoming edge cut**: All `r_{j→i}=0` → child doesn't depend on parent embedding |
| T-09 | **Outgoing edge preserve**: `r_{j→i}=0` for incoming but `r_{i→k}=1` for outgoing — child output unaffected |
| T-10 | **Node deletion**: `o_i=0` + all edges cut → node i completely removed |
| T-11 | **Edge-only ablation**: Single edge cut → other edges still functional |
| T-12 | **Invalid intervention**: Missing required fields raises ValueError |

## Path Invariance Tests

| T-13 | **Denoiser invariance**: Same x_t, f_i changed, o_i=0 → ε identical (autograd check) |
| T-14 | **Trajectory invariance DDIM**: Same x_T, DDIM η=0, o_i=0 → identical final sample |
| T-15 | **Trajectory invariance DDPM**: Same x_T + same noise trace, o_i=0 → identical sample |
| T-16 | **Same NoiseTrace identity**: Same model + condition + NoiseTrace → exact output match |

## Checkpoint Tests

| T-17 | **Config roundtrip**: Save model with DAG config → reload → graph edges preserved |
| T-18 | **Resume equivalence**: Train N steps, save, resume, train N more = train 2N steps |
| T-19 | **RNG restore**: Same seed + checkpoint → same next batch after resume |

## Baseline Tests

| T-20 | **CF-DiT null ≠ class 0**: Dropped sample with null token gets different embedding from class 0 sample |
| T-21 | **DiTBlock adaLN-Zero**: Zero-initialized output layer → initial output ≈ 0 |

## Dataset Tests

| T-22 | **Split disjointness**: Train ∩ test = ∅ |
| T-23 | **Held-out combination**: Specified held-out combinations not in train set |
| T-24 | **Split reproducibility**: Same seed → same split indices |

## Metric Tests

| T-25 | **No-op identity**: Same sample compared to itself → edit rate = 0, leakage = I matrix |
| T-26 | **New ≠ old**: Intervention sampling never yields identical factor value |
| T-27 | **Leakage matrix shape**: K×K, diagonal ≥ 0, bounded in [0,1] |
| T-28 | **Oracle perfect case**: Synthetic oracle returns ground truth on training data |

## Acceptance Criteria Before GPU Training

1. All 28 tests pass
2. Path invariance: max |ε(f_i) - ε(f_i')| with o_i=0 < 1e-5
3. Trajectory invariance (DDIM): max |X_0(f_i) - X_0(f_i')| with o_i=0 < 1e-3
4. Same NoiseTrace identity: max |X_0 - X_0'| < 1e-6
5. Checkpoint roundtrip: load(save(model)) produces identical output
