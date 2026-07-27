# Factor-Gated Routing (FGR)

**An Architectural Primitive for Verifiable Controllable Diffusion**

[Paper] • [arXiv] • [Checkpoints (coming soon)]

Factor-Gated Routing (FGR) introduces per-factor architectural streams
with directed-acyclic-graph (DAG) cross-stream attention and
inference-time intervention gating. FGR's gate mechanism is an
architectural analog of Pearl's do-operator: setting `gate_i = 0`
isolates factor `i`'s contribution to the generative process, enabling
verifiable controllable generation and factor-level model auditing.

For theoretical motivation, see [`MATH_NOTES.md`](MATH_NOTES.md).

## Structure

```
src/
├── model.py          FGRDiT, FGRStream
├── baselines.py      SDiT, EncDiff, MMDiT-k, CoInD, CF-DiT
├── train.py          Training (6 models × 2 datasets)
├── evaluate.py       Evaluation (conditional accuracy, oracle change, gate-sweep)
├── diffusion.py      DDPM sampling and noise schedules
├── oracle.py         Oracle factor classifier
├── config.py         Model config, path resolution
├── dataset.py        DSprites, 3DShapes loaders
├── utils.py          DiTBlock, CrossAttnBlock, AdaLN
└── registry.py       Shared model factory

configs/
├── dsprites.yaml     dSprites experiment config
└── shapes3d.yaml     3DShapes experiment config

tests/                21 unit tests (model, baselines, diffusion, dataset)
scripts/              Batch run scripts
```

## Setup

### Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.7 (CUDA 12.8)
- NumPy, h5py, Pillow

```bash
git clone <repo-url>
cd gauge-sensitive-inverse-generation
pip install -e .
```

### Dataset

Download the datasets and set environment variables:

```bash
# dSprites (737K binary shape images, 3 factors)
# https://github.com/deepmind/dsprites-dataset
export DSPRITES_PATH=/path/to/dsprites.npz

# 3DShapes (480K colored objects, 6 factors)
# https://github.com/deepmind/3d-shapes
export SHAPES3D_PATH=/path/to/3dshapes.h5

# Output directory for checkpoints, logs, and evaluation results
export FGR_OUTPUT_DIR=output
mkdir -p $FGR_OUTPUT_DIR
```

Or copy `.env.example` to `.env` and edit:
```bash
cp .env.example .env
# edit .env with your paths
source .env
```

### Oracle classifier

The evaluation pipeline requires a trained oracle classifier to measure
factor-level changes. Train it once:

```python
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.oracle import OracleClassifier
from src.dataset import DSpritesDataset

device = torch.device("cuda")
factor_sizes = (3, 6, 40)  # shape, scale, orientation
oracle = OracleClassifier(factor_sizes, in_channels=1).to(device)

ds = DSpritesDataset("/path/to/dsprites.npz", split="train")
loader = DataLoader(ds, batch_size=256, shuffle=True)
opt = AdamW(oracle.parameters(), lr=1e-3)

for epoch in range(10):
    for batch in loader:
        img, factors = batch["image"].to(device), batch["factors"].to(device)
        preds = oracle(img)
        loss = sum(F.cross_entropy(p, factors[:, i]) for i, p in enumerate(preds))
        opt.zero_grad(); loss.backward(); opt.step()

torch.save(oracle.state_dict(), "output/oracle.pt")
```

## Usage

### Training

Single model:
```bash
python -m src.train \
  --model FGR --dataset dsprites \
  --steps 400000 --batch-size 128 \
  --output-dir output/fgr_seed42 \
  --seed 42
```

All 6 models sequentially:
```bash
bash scripts/run_full_experiment.sh dsprites 400000 42
```

### Ablations

```bash
# Full cross-attention (all-to-all, no DAG topology)
python -m src.train --model FGR --ablation full_ca ...

# No inter-stream cross-attention (independent streams)
python -m src.train --model FGR --ablation no_inter_stream ...
```

### Evaluation

```bash
export FGR_ORACLE_PATH=output/oracle.pt

python -m src.evaluate \
  --model FGR --dataset dsprites \
  --checkpoint output/fgr_seed42/FGR_ema_final.pt \
  --n-samples 256 --n-steps 200 \
  --gate-sweep \
  --output-dir output/eval
```

The evaluation reports:
- **Conditional accuracy**: how often the oracle predicts the correct factor class from generated images
- **Oracle change**: fraction of samples where factor prediction changes under intervention. FGR with `gate=0` should show low change on the intervened factor.
- **Gate sweep** (`--gate-sweep`): tests monotonicity — oracle change should decrease as gate decreases from 1.0 → 0.5 → 0.0
- **Non-intervention stability**: fraction of samples with pixel change < threshold under intervention

## Architecture

```
Input (noisy image x_t)
    │
    ├─── patch_embed ──→ shared tokens [B, N, dim]
    │                          │
    ├── Factor 0 (shape)  ───→ Stream 0 ──→ × gate₀ ──→ s₀
    ├── Factor 1 (scale)  ───→ Stream 1 ──→ × gate₁ ──→ s₁
    └── Factor 2 (orient.) ──→ Stream 2 ──→ × gate₂ ──→ s₂
                                                          │
                                          Σ sᵢ → LayerNorm → Linear → pixels
```

Each stream is an independent DiT block receiving its factor's embedding
via `AdaLN` modulation. Cross-stream attention follows a DAG topology
(configurable via `dag_edges`). The gate `∈ [0, 1]` multiplicatively
scales each stream's output before summation.

## Baselines

| Model | Params | Architecture |
|-------|--------|-------------|
| **FGR** | 13.24M | Per-factor DiT streams + DAG cross-attn + gating |
| SDiT | 13.24M | Single-stream DiT, factor embeddings summed |
| EncDiff | 13.25M | Cross-attn conditioning (no AdaLN modulation) |
| MMDiT-k | 18.01M | Multi-stream DiT, all-to-all synchronous cross-attn |
| CoInD | 13.24M | Per-stream independent DiT blocks, no cross-attn |
| CF-DiT | 13.24M | SDiT with classifier-free guidance dropout (p=0.1) |

## Configuration Reference

| Env variable | Required | Description |
|-------------|----------|-------------|
| `DSPRITES_PATH` | Yes | Path to dSprites `.npz` file |
| `SHAPES3D_PATH` | Yes | Path to 3DShapes `.h5` file |
| `FGR_OUTPUT_DIR` | No (`output/`) | Base directory for checkpoints/logs/results |
| `FGR_ORACLE_PATH` | No (`output/oracle.pt`) | Path to oracle checkpoint |

## Citation

```bibtex
@inproceedings{fgr2027,
  title={Factor-Gated Routing: An Architectural Primitive for Verifiable Controllable Diffusion},
  author={...},
  booktitle={...},
  year={2027}
}
```

## License

MIT
