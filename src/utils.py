import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import torch
except Exception:
    torch = None

def set_global_seed(seed: int) -> None:
    # keeps random behavior reproducible
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    # makes sure output folders exist
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    # writes json in a readable format
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


