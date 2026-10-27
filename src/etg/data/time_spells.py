"""Map timestamps to time-spell indices t ∈ {0..T-1}.

A "time-spell" is a contiguous temporal bucket (e.g., a calendar month).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TimeSpellIndex:
    """Uniform bucketing between [t_min, t_max] into T spells."""

    t_min: datetime
    t_max: datetime
    T: int

    def __post_init__(self) -> None:
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be strictly greater than t_min")
        if self.T < 1:
            raise ValueError("T must be >= 1")
        self._total = (self.t_max - self.t_min).total_seconds()
        self._step = self._total / self.T

    def assign(self, ts: datetime) -> int:
        """Return the spell index for timestamp `ts` (clamped to [0, T-1])."""
        if ts <= self.t_min:
            return 0
        if ts >= self.t_max:
            return self.T - 1
        delta = (ts - self.t_min).total_seconds()
        idx = int(delta // self._step)
        return max(0, min(idx, self.T - 1))

    @classmethod
    def from_posts(cls, posts, T: int) -> "TimeSpellIndex":
        ts_list = [p.timestamp for p in posts]
        return cls(t_min=min(ts_list), t_max=max(ts_list), T=T)
