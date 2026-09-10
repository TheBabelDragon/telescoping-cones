from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cone import Collapse, TelescopingCone

LAYER_STEMS = ("perception", "structure", "relation", "salience", "semantic")


def layer_name(k: int, depth: int) -> str:
    if k == depth - 1:
        return "state"
    if k < len(LAYER_STEMS):
        return LAYER_STEMS[k]
    return f"cone-{k}"


def _sigmoid(x: float) -> float:
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


def next_shrink(entropy: float, count: int, adaptive: bool) -> float:
    """r_{k+1} = r_k · σ(f(X_k)). High entropy keeps the next cone open."""
    if not adaptive:
        return 0.58
    hmax = math_log(max(count, 2))
    hnorm = min(1.0, max(0.0, entropy / hmax))
    opened = _sigmoid(hnorm * 6.0 - 2.4)
    return float(min(0.88, max(0.34, 0.38 + 0.5 * opened)))


def math_log(x: float) -> float:
    return float(np.log(x))


@dataclass
class Layer(Collapse):
    k: int
    name: str
    aperture: float
    radius: float
    shrink: float
    cone: TelescopingCone


@dataclass
class Hierarchy:
    layers: list[Layer]
    axis: np.ndarray


def build_hierarchy(
    axis,
    aperture0: float,
    radius0: float,
    depth: int,
    temperature: float,
    adaptive: bool,
    points,
    features,
    metric: str = "angular",
    apex=None,
) -> Hierarchy:
    depth = int(min(6, max(2, round(depth))))
    layers: list[Layer] = []
    aperture = float(aperture0)
    radius = float(radius0)
    axis = np.asarray(axis, dtype=np.float64)
    for k in range(depth):
        cone = TelescopingCone(
            center=axis,
            radius=radius,
            depth=k,
            aperture=aperture,
            temperature=temperature,
            apex=apex,
            metric=metric,
        )
        collapsed = cone.collapse(features, points)
        shrink = 1.0 if k == depth - 1 else next_shrink(collapsed.entropy, collapsed.count, adaptive)
        layers.append(
            Layer(
                state=collapsed.state,
                entropy=collapsed.entropy,
                mass=collapsed.mass,
                count=collapsed.count,
                weights=collapsed.weights,
                k=k,
                name=layer_name(k, depth),
                aperture=aperture,
                radius=radius,
                shrink=shrink,
                cone=cone,
            )
        )
        aperture *= shrink
        radius *= shrink
    return Hierarchy(layers, axis)
