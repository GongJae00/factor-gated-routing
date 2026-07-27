from fgr.model import FGRDiT
from fgr.baselines import build_baseline

MODEL_REGISTRY = {
    "FGR": FGRDiT,
    "SDiT": lambda cfg: build_baseline("SDiT", cfg),
    "EncDiff": lambda cfg: build_baseline("EncDiff", cfg),
    "MMDiT-k": lambda cfg: build_baseline("MMDiT-k", cfg),
    "CoInD": lambda cfg: build_baseline("CoInD", cfg),
    "CF-DiT": lambda cfg: build_baseline("CF-DiT", cfg),
}
