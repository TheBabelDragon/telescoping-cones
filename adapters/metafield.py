"""MetaField adapter. Core never depends on MetaField."""

from __future__ import annotations

from telescoping_cones import TelescopingCone, build_hierarchy, encode_cone_state


def from_world_state(world, axis, aperture=0.5, depth=4, temperature=1.0, adaptive=True):
    """Collapse a MetaField world snapshot through telescoping cones.

    `world` is expected to expose `.positions` (N,3) and `.features` (N,D).
    """
    points = world.positions
    features = world.features
    hierarchy = build_hierarchy(
        axis=axis,
        aperture0=aperture,
        radius0=aperture * 1.15,
        depth=depth,
        temperature=temperature,
        adaptive=adaptive,
        points=points,
        features=features,
    )
    return encode_cone_state("metafield", axis, hierarchy, hierarchy.layers[-1].state, [], getattr(world, "tick", 0))


def cone_from_state(state):
    return TelescopingCone(
        center=state.position,
        radius=getattr(state, "radius", 1.0),
        depth=getattr(state, "depth", 6),
        aperture=getattr(state, "aperture", 0.35),
    )
