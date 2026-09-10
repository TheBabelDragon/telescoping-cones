import math

import numpy as np

from telescoping_cones import (
    ConeMemory,
    TelescopingCone,
    build_hierarchy,
    cap_radius_on_plane,
    cone_affinity,
    cone_attention,
    encode_cone_state,
    fibonacci_sphere,
    make_projection,
    next_shrink,
    route_cones,
    stereographic,
)


def test_nested_radii_decrease():
    points = fibonacci_sphere(48)
    features = np.concatenate([points, np.ones((48, 5))], axis=1)
    h = build_hierarchy(
        axis=[0, 0, 1],
        aperture0=0.8,
        radius0=0.9,
        depth=4,
        temperature=1.0,
        adaptive=False,
        points=points,
        features=features,
    )
    for a, b in zip(h.layers, h.layers[1:]):
        assert b.radius < a.radius
        assert b.aperture < a.aperture
    assert h.layers[-1].name == "state"
    assert h.layers[0].name == "perception"


def test_contains_is_subset():
    points = fibonacci_sphere(64)
    wide = TelescopingCone(center=[0, 1, 0], radius=0.9, aperture=0.8)
    narrow = TelescopingCone(center=[0, 1, 0], radius=0.4, aperture=0.35)
    assert np.all(~narrow.contains(points) | wide.contains(points))


def test_stereographic_cap_identity():
    alpha = 0.6
    t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    ring = np.stack([np.sin(alpha) * np.cos(t), np.sin(alpha) * np.sin(t), np.full_like(t, np.cos(alpha))], axis=1)
    uv = stereographic(ring, [0, 0, 1])
    rs = np.linalg.norm(uv, axis=1)
    assert np.allclose(rs, cap_radius_on_plane(alpha), atol=1e-6)


def test_attention_rows_sum_to_one():
    points = fibonacci_sphere(12)
    features = np.concatenate([points, np.linspace(0, 1, 12)[:, None].repeat(5, axis=1)], axis=1)
    A = cone_affinity(points, [0, 0, 0], 0.4, 0.5)
    Wq = make_projection(8, 1112)
    Wk = make_projection(8, 2213)
    Wv = make_projection(8, 3324)
    _, W = cone_attention(features, A, 1.1, 0.9, Wq, Wk, Wv)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-6)


def test_uncertainty_opens_relative_to_confidence():
    assert next_shrink(2.4, 40, True) > next_shrink(0.2, 40, True)


def test_memory_ema():
    m = ConeMemory(3, alpha=0.5)
    m.update([1, 0, 0], [0, 0, 1], 0.4, 0.4)
    m.update([0, 1, 0], [0, 1, 0], 0.3, 0.3)
    assert m.state[0] > 0 and m.state[1] > 0
    assert len(m.traces) == 2


def test_aurora_routing():
    points = fibonacci_sphere(48)
    features = np.concatenate([points, np.ones((48, 5))], axis=1)
    tree = route_cones([0.4, 0.5, 0.7], 0.7, 0.8, points, features)
    assert tree.root.id == "A"
    assert len(tree.flat) >= 1


def test_protocol():
    points = fibonacci_sphere(16)
    features = np.concatenate([points, np.zeros((16, 5))], axis=1)
    h = build_hierarchy([0, 0, 1], 0.5, 0.6, 3, 1.0, False, points, features)
    proto = encode_cone_state("sphere", [0, 0, 1], h, np.zeros(8), [], 4)
    assert proto["version"] == "cone-state-v0.1"
    assert len(proto["hash"]) == 8
    assert len(proto["radii"]) == 3
