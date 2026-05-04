"""Lightweight disk cache keyed by a config fingerprint + call-site key."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable

import torch


def _key_hash(parts: tuple[Any, ...]) -> str:
    raw = pickle.dumps(parts)
    return hashlib.sha256(raw).hexdigest()[:16]


class DiskCache:
    """Simple content-addressed pickle/tensor cache."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str, ext: str = "pkl") -> Path:
        return self.cache_dir / f"{key}.{ext}"

    def get(self, key: str) -> Any | None:
        p_pt = self.path(key, "pt")
        if p_pt.exists():
            return torch.load(p_pt, map_location="cpu", weights_only=False)
        p = self.path(key, "pkl")
        if p.exists():
            with p.open("rb") as f:
                return pickle.load(f)
        return None

    def put(self, key: str, obj: Any) -> None:
        # torch tensors / state dicts → .pt; everything else → .pkl
        if isinstance(obj, (torch.Tensor, dict)) and self._has_tensors(obj):
            torch.save(obj, self.path(key, "pt"))
        else:
            with self.path(key, "pkl").open("wb") as f:
                pickle.dump(obj, f)

    @staticmethod
    def _has_tensors(obj: Any) -> bool:
        if isinstance(obj, torch.Tensor):
            return True
        if isinstance(obj, dict):
            return any(isinstance(v, torch.Tensor) for v in obj.values())
        return False

    def memoize(self, namespace: str, config_hash: str) -> Callable:
        """Decorator: key = (namespace, config_hash, args, kwargs)."""

        def deco(fn: Callable) -> Callable:
            def wrapped(*args, **kwargs):
                key = _key_hash((namespace, config_hash, args, kwargs))
                hit = self.get(key)
                if hit is not None:
                    return hit
                out = fn(*args, **kwargs)
                self.put(key, out)
                return out

            wrapped.__name__ = fn.__name__
            return wrapped

        return deco
