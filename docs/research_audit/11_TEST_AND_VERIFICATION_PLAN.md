# 11 — Test and Verification Plan (v2)

**Minimum 35 tests required for correctness confidence before GPU experiments.**

## Graph Tests

| T-01 | **Cycle detection in DAG mode**: edges `[(0,1),(1,2),(2,0)]` → ValueError |
| T-02 | **Invalid node**: Edge `[(0, 10)]` with 3 factors → ValueError |
| T-03 | **Duplicate edge**: `[(0,1),(0,1)]` → ValueError or dedup warning |
| T-04 | **Self-loop**: `[(0,0)]` → ValueError |
| T-05 | **DENSE_DIRECTED accepts 2-edges**: `[(0,1),(1,0)]` in DENSE_DIRECTED mode → no error (not a DAG) |
| T-06 | **Node permutation consistency**: Renumbering nodes 0↔1 + permuting embeddings → identical output |
| T-07 | **Graph type roundtrip**: Save model with DAG config → reload → edges preserved in correct order |

## Intervention Tests

| T-08 | **observational baseline**: No intervention → output matches normal forward |
| T-09 | **direct_output_ablation**: o_i=0 → P_i·Norm_i(h_i) contribution = 0 (verified by direct measurement) |
| T-10 | **factor_source_cut invariance**: All pathways from e_i to output are cut → output invariant to f_i change (Path Non-Interference Theorem) |
| T-11 | **outgoing_edge_preserve**: incoming_cut on node i, outgoing_preserve on node i → descendants at child k receive message from i |
| T-12 | **node_deletion zero output**: o_i=0 + r_{*→i}=0 + r_{i→*}=0 → node i absent from computation |
| T-13 | **edge_ablation specificity**: Cut single edge j→i → other edges j→k still functional |
| T-14 | **neural_graph_surgery semantics**: incoming cut on i + f_i=v' injected + outgoing to children preserved → children reflect v' |
| T-15 | **invalid intervention spec**: Missing required field for mode → ValueError |
| T-16 | **condition_mask**: f_i replaced with null token → embedding differs from all valid classes |

## Path Invariance Tests

| T-17 | **Denoiser invariance under factor_source_cut**: Same x_t, factor_source_cut on i, f_i changed → |ε - ε'| < tolerance |
| T-18 | **Trajectory invariance DDIM**: Same x_T, DDIM η=0, factor_source_cut on i → identical final sample |
| T-19 | **Trajectory invariance DDPM**: Same x_T + same counter-seed per step, factor_source_cut on i → identical sample |
| T-20 | **NoiseTrace identity DDIM**: Same model + condition + DDIM NoiseTrace → exact output match |
| T-21 | **NoiseTrace identity DDPM**: Same model + condition + DDPM counter-seed trace → exact output match |
| T-22 | **Independent noise NOT identical**: Different NoiseTrace → different outputs (confirms randomness coupling works) |

## Checkpoint Tests

| T-23 | **Config roundtrip**: Save FGR with DAG → reload → edges preserved, output matches |
| T-24 | **Full checkpoint**: model, optimizer, scheduler, scaler, RNG, config, graph, step saved and restorable |
| T-25 | **RNG restore**: Same seed + checkpoint → next training batch identical after resume |

## Baseline Tests

| T-26 | **CF-DiT null ≠ class 0**: Null token for factor i produces different embedding from class 0 for factor i |
| T-27 | **CF-DiT per-factor dropout**: Each factor independently dropped, not all-or-nothing |
| T-28 | **DiTBlock adaLN-Zero identity**: Zero-initialized output layer → initial residual contribution ≈ 0, block output ≈ input (residual identity) |
| T-29 | **IndependentStreamDiT ≠ CoInD**: Verify no Fisher divergence loss (name change validated) |

## Dataset Tests

| T-30 | **Split disjointness**: Train ∩ validation = ∅, train ∩ test = ∅ |
| T-31 | **Held-out combination**: Specified held-out (f_1, f_2) tuples absent from train set |
| T-32 | **Split reproducibility**: Same seed → identical train/test indices |
| T-33 | **Lazy loading memory**: 3DShapes loader peak RAM < 1 GB (vs current ~22 GB) |

## Metric Tests

| T-34 | **No-op zero change**: Same sample compared to itself → OffTargetChange[i,j] = 0 for all i,j |
| T-35 | **New ≠ old**: Offset-based sampling: new_val randomly offset from old → never equal |
| T-36 | **TargetSuccess shape**: K-vector, each entry in [0,1] |
| T-37 | **OffTargetChange matrix shape**: K×K, off-diagonal entries in [0,1], diagonal = TargetSuccess complement OR separate |
| T-38 | **Oracle fixture test**: Deterministic synthetic oracle (returns ground-truth from labels) → metric implementation verified |

## Acceptance Criteria Before GPU Training

1. All 38 tests pass
2. Path invariance (factor_source_cut): max |ε(f_i) - ε(f_i')| < 1e-5 (fp32), < 1e-3 (fp16/bf16)
3. Trajectory invariance (DDIM): max |X_0(f_i) - X_0(f_i')| < 1e-4 (fp32)
4. NoiseTrace identity: max |X_0 - X_0'| < 1e-6 (fp32)
5. Checkpoint roundtrip: load(save(model)) produces identical output
6. Tolerance annotations: per-dtype, per-device. bf16 trajectory tolerance appropriately larger than fp32.
