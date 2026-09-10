from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import angular_gate, geodesic, normalize, radial_gate


@dataclass
class Collapse:
    state: np.ndarray
    entropy: float
    mass: float
    count: int
    weights: np.ndarray


class TelescopingCone:
    """Geometric inclusion + weighting. Neural code does not live here."""

    def __init__(
        self,
        center,
        radius: float,
        depth: int = 1,
        aperture: float = 0.35,
        temperature: float = 1.0,
        apex=None,
        metric: str = "angular",
        sigma: float | None = None,
    ):
        self.center = normalize(np.asarray(center, dtype=np.float64))
        self.radius = float(radius)
        self.depth = int(depth)
        self.aperture = float(aperture)
        self.temperature = float(temperature)
        self.apex = np.zeros(3) if apex is None else np.asarray(apex, dtype=np.float64)
        self.metric = metric
        self.sigma = float(sigma) if sigma is not None else max(self.radius * 0.55, 1e-4)

    def angle_to(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return geodesic(points - self.apex, self.center)

    def distance(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if self.metric == "angular":
            return self.angle_to(points)
        return np.linalg.norm(points - self.apex, axis=-1)

    def contains(self, points) -> np.ndarray:
        return (self.distance(points) < self.radius) & (self.angle_to(points) < self.aperture)

    def weights(self, points) -> np.ndarray:
        d = self.distance(points)
        th = self.angle_to(points)
        return radial_gate(d, self.sigma, self.temperature) * angular_gate(th, self.aperture)

    def collapse(self, features, points) -> Collapse:
        features = np.asarray(features, dtype=np.float64)
        w = self.weights(points)
        mass = float(w.sum())
        if mass < 1e-12 or features.size == 0:
            dim = features.shape[1] if features.ndim == 2 else 0
            return Collapse(np.zeros(dim), 0.0, 0.0, 0, w)
        p = w / mass
        nz = p[p > 1e-15]
        entropy = float(-(nz * np.log(nz)).sum())
        state = p @ features
        count = int((w > 1e-4).sum())
        return Collapse(state, entropy, mass, count, w)

    def transform(self, observations, geometry) -> Collapse:
        return self.collapse(observations, geometry)
