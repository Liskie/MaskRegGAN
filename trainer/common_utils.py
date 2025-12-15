import time
from typing import Dict

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.transforms import ToPILImage


class _Timer:
    """Simple context manager accumulating elapsed wall time into a dict."""

    def __init__(self, name: str, prof_dict: Dict[str, float]):
        self.name = name
        self.prof = prof_dict

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.perf_counter()
        self.prof[self.name] += (t1 - self.t0)


class LogRange:
    """Utility for printing tensor/image ranges while debugging augmentations."""

    def __init__(self, tag: str):
        self.tag = tag

    def __call__(self, x):
        try:
            if isinstance(x, torch.Tensor):
                vmin = float(x.min().item())
                vmax = float(x.max().item())
                dtype = str(x.dtype)
                shape = tuple(x.shape)
            elif isinstance(x, np.ndarray):
                vmin = float(np.min(x))
                vmax = float(np.max(x))
                dtype = str(x.dtype)
                shape = x.shape
            elif isinstance(x, PILImage.Image):
                arr = np.array(x)
                vmin = float(arr.min())
                vmax = float(arr.max())
                dtype = f"PIL[{x.mode}]/{arr.dtype}"
                shape = arr.shape
            else:
                dtype = type(x).__name__
                shape = getattr(x, 'size', None)
                vmin = float('nan')
                vmax = float('nan')
            print(f"[{self.tag}] type={type(x).__name__} shape={shape} dtype={dtype} min={vmin:.6g} max={vmax:.6g}",
                  flush=True)
        except Exception as e:
            print(f"[{self.tag}] (failed to compute stats: {e})", flush=True)
        return x


class ToPILWithLog:
    """Wrapper around torchvision's ToPILImage that logs value ranges before/after."""

    def __init__(self, tag: str = "ToPILImage"):
        self._op = ToPILImage()
        self._before = LogRange(f"before {tag}")
        self._after = LogRange(f"after {tag}")

    def __call__(self, x):
        self._before(x)
        y = self._op(x)
        self._after(y)
        return y
