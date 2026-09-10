from __future__ import annotations

import math

import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a)
    b = normalize(b)
    d = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(d)


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    theta = 2.0 * math.pi * i / golden
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def tangent_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = normalize(axis)
    helper = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = normalize(np.cross(helper, axis))
    t2 = np.cross(axis, t1)
    return t1, t2


def rotate_to_pole(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    axis = normalize(axis)
    c = float(axis[2])
    if c > 0.999999:
        return points.copy()
    if c < -0.999999:
        out = points.copy()
        out[:, 2] *= -1.0
        return out
    v = np.array([-axis[1], axis[0], 0.0])
    k = 1.0 / (1.0 + c)
    vx = np.cross(v, points)
    vxx = np.cross(v, vx)
    return points + vx + vxx * k


def stereographic(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """South-pole projection after rotating `axis` to +Z. Focus at the origin."""
    p = rotate_to_pole(points, axis)
    denom = np.where(np.abs(1.0 + p[:, 2]) < 1e-9, 1e-9, 1.0 + p[:, 2])
    return np.stack([p[:, 0] / denom, p[:, 1] / denom], axis=1)


def cap_radius_on_plane(aperture: float | np.ndarray) -> np.ndarray:
    return np.tan(np.asarray(aperture, dtype=np.float64) / 2.0)


def dodecahedron_faces() -> np.ndarray:
    """Face centers of a regular dodecahedron = icosahedron vertices."""
    verts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            verts.append([0.0, s1, s2 * PHI])
            verts.append([s1, s2 * PHI, 0.0])
            verts.append([s1 * PHI, 0.0, s2])
    return normalize(np.asarray(verts, dtype=np.float64))


def angular_gate(theta: np.ndarray, aperture: float) -> np.ndarray:
    a = max(float(aperture), 1e-6)
    return np.exp(-(theta ** 2) / (2.0 * a * a))


def radial_gate(distance: np.ndarray, sigma: float, temperature: float = 1.0) -> np.ndarray:
    s = max(float(sigma), 1e-6)
    t = max(float(temperature), 1e-4)
    return np.exp(-(distance ** 2) / (2.0 * s * s * t))


def offset_axis(axis: np.ndarray, tangent: np.ndarray, angle: float) -> np.ndarray:
    return normalize(normalize(axis) * math.cos(angle) + normalize(tangent) * math.sin(angle))


def principal_tangent(axis: np.ndarray, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    t1, t2 = tangent_basis(axis)
    u = points @ t1
    v = points @ t2
    s11 = float(np.sum(weights * u * u))
    s12 = float(np.sum(weights * u * v))
    s22 = float(np.sum(weights * v * v))
    ang = 0.5 * math.atan2(2.0 * s12, 1.0 if s11 == s22 and s12 == 0.0 else (s11 - s22))
    return normalize(t1 * math.cos(ang) + t2 * math.sin(ang))
