"""Telescoping cones — geometric attention operator.

Geometry is independent of the neural backend. The same cone mathematics
run in NumPy, optional PyTorch, MetaField, Canonsphere, Aurora routing,
and optical-body observations.
"""

from .attention import cone_affinity, cone_attention, make_projection
from .cone import Collapse, TelescopingCone
from .geometry import (
    cap_radius_on_plane,
    dodecahedron_faces,
    fibonacci_sphere,
    geodesic,
    stereographic,
    tangent_basis,
)
from .hierarchy import build_hierarchy, layer_name, next_shrink
from .memory import ConeMemory
from .protocol import PROTOCOL_VERSION, encode_cone_state
from .routing import route_cones

__all__ = [
    "Collapse",
    "ConeMemory",
    "PROTOCOL_VERSION",
    "TelescopingCone",
    "build_hierarchy",
    "cap_radius_on_plane",
    "cone_affinity",
    "cone_attention",
    "dodecahedron_faces",
    "encode_cone_state",
    "fibonacci_sphere",
    "geodesic",
    "layer_name",
    "make_projection",
    "next_shrink",
    "route_cones",
    "stereographic",
    "tangent_basis",
]

try:
    from .torch_attention import ConeAttention
except ImportError:
    ConeAttention = None  # type: ignore[misc, assignment]
else:
    __all__.append("ConeAttention")
