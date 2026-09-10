from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MemoryTrace:
    axis: np.ndarray
    radius: float
    aperture: float
    state: np.ndarray
    t: int


@dataclass
class ConeMemory:
    """M_{t+1} = α M_t + (1-α) C_t

    Stores the geometric narrowing that produced the conclusion.
    """

    dim: int
    alpha: float = 0.86
    max_traces: int = 12
    state: np.ndarray = field(init=False)
    traces: list[MemoryTrace] = field(default_factory=list)
    tick: int = 0

    def __post_init__(self) -> None:
        self.state = np.zeros(self.dim, dtype=np.float64)

    def update(self, collapsed, axis, radius: float, aperture: float, alpha: float | None = None) -> None:
        collapsed = np.asarray(collapsed, dtype=np.float64)
        a = self.alpha if alpha is None else alpha
        if self.state.shape != collapsed.shape:
            self.state = collapsed.copy()
        else:
            self.state = a * self.state + (1.0 - a) * collapsed
        self.tick += 1
        self.traces.insert(
            0,
            MemoryTrace(np.asarray(axis, dtype=np.float64).copy(), float(radius), float(aperture), collapsed.copy(), self.tick),
        )
        if len(self.traces) > self.max_traces:
            self.traces.pop()
