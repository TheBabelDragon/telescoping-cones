from __future__ import annotations

import math

import numpy as np

from .geometry import angular_gate, normalize


def make_projection(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(dim):
        row = rng.standard_normal(dim)
        for prev in rows:
            row = row - float(row @ prev) * prev
        n = np.linalg.norm(row)
        rows.append(row / max(n, 1e-12))
    return np.stack(rows, axis=0)


def cone_affinity(points, apex, sigma: float, aperture: float) -> np.ndarray:
    """A_ij = exp(-d(p_i, p_j)^2 / (2σ²)) · g(θ_ij)."""
    P = np.asarray(points, dtype=np.float64)
    apex = np.asarray(apex, dtype=np.float64)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    V = normalize(P - apex)
    cos = np.clip(V @ V.T, -1.0, 1.0)
    theta = np.arccos(cos)
    g = angular_gate(theta, aperture)
    s = max(float(sigma), 1e-6)
    return np.exp(-d2 / (2.0 * s * s)) * g


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)


def cone_attention(features, affinity, lam: float, temperature: float, Wq, Wk, Wv):
    """Attention(Q,K,V) = softmax(QKᵀ/√d + λ A^cone) V."""
    X = np.asarray(features, dtype=np.float64)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d = Q.shape[1]
    scores = (Q @ K.T) / math.sqrt(d) + lam * affinity
    T = max(float(temperature), 1e-4)
    W = _softmax(scores / T)
    return W @ V, W
