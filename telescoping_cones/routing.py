from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cone import TelescopingCone
from .geometry import offset_axis, principal_tangent
from .hierarchy import next_shrink

ENTROPY_EXPAND = 1.15
MIN_MASS = 0.35


@dataclass
class Worker:
    id: str
    axis: np.ndarray
    aperture: float
    radius: float
    mass: float
    entropy: float
    count: int
    expanded: bool
    children: list["Worker"] = field(default_factory=list)


@dataclass
class RoutingTree:
    root: Worker
    flat: list[Worker]


def _walk(
    ident: str,
    axis,
    aperture: float,
    radius: float,
    points,
    features,
    depth_left: int,
    temperature: float,
) -> Worker:
    cone = TelescopingCone(center=axis, radius=radius, aperture=aperture, temperature=temperature)
    c = cone.collapse(features, points)
    should = depth_left > 0 and c.entropy > ENTROPY_EXPAND and c.mass > MIN_MASS and c.count > 6
    node = Worker(ident, np.asarray(axis, dtype=np.float64), aperture, radius, c.mass, c.entropy, c.count, should, [])
    if not should:
        return node
    tangent = principal_tangent(axis, np.asarray(points, dtype=np.float64), c.weights)
    split = min(aperture * 0.55, 0.32)
    shrink = next_shrink(c.entropy, c.count, True)
    child_a, child_r = aperture * shrink, radius * shrink
    node.children = [
        _walk(ident + "1", offset_axis(axis, tangent, split), child_a, child_r, points, features, depth_left - 1, temperature),
        _walk(ident + "2", offset_axis(axis, tangent, -split), child_a, child_r, points, features, depth_left - 1, temperature),
    ]
    return node


def _flatten(node: Worker, acc: list[Worker]) -> None:
    acc.append(node)
    for child in node.children:
        _flatten(child, acc)


def route_cones(axis, aperture, radius, points, features, temperature: float = 1.0, max_depth: int = 2) -> RoutingTree:
    """Aurora owns routing, not the cone transform. Expand only where entropy deserves it."""
    root = _walk("A", axis, aperture, radius, points, features, max_depth, temperature)
    flat: list[Worker] = []
    _flatten(root, flat)
    return RoutingTree(root, flat)
