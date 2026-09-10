# telescoping-cones

Geometric attention operator.

Information enters through a wide cross-section, is progressively compressed by nested cones, and exits as a narrow state. Geometry is independent of the neural implementation.

```
raw field
    │
    ▼
Cone 0  perception
Cone 1  structure
Cone 2  relation
Cone 3  state
    │
    ▼
memory
```

Live lab (interactive Canonsphere + cone attention): this repository is the reusable module. The operator is not a renamed Transformer layer. Geometry actually modifies attention.

```
Attention(Q, K, V) = softmax( QKᵀ/√d  +  λ A^cone ) V
```

with

```
A_ij^cone = exp( -d(p_i, p_j)² / 2σ_k² ) · g(θ_ij)
```

and nested radii

```
C_k = { x : d(x, o) < r_k }
r_{k+1} = r_k · σ(f(X_k))
```

Uncertainty opens the next cone. Confidence collapses it.

## Install

```bash
pip install -e .
```

NumPy is required. PyTorch is optional (`ConeAttention`).

```python
from telescoping_cones import TelescopingCone, build_hierarchy, cone_attention

cone = TelescopingCone(
    center=state_axis,
    radius=1.0,
    depth=6,
    aperture=0.35,
)
collapsed = cone.transform(observations, geometry)
```

```python
from telescoping_cones import ConeAttention  # requires torch
layer = ConeAttention(dim=32)
y = layer(x, geometry=positions)
```

## Package

```
telescoping_cones/
  geometry.py      vectors, geodesic, stereograph, dodecahedron
  cone.py          TelescopingCone.contains / weights / collapse
  attention.py     A^cone bias + softmax attention
  hierarchy.py     nested adaptive radii
  routing.py       Aurora recursive subdivision
  memory.py        M_{t+1} = α M_t + (1-α) C_t
  protocol.py      cone-state-v0.1
  torch_attention.py   optional nn.Module
js/                same mathematics for Canonsphere / browser
```

## Where it plugs in

```
        TELESCOPING CONES
                │
   ┌────────────┼────────────┐
   ▼            ▼            ▼
MetaField     Aurora      Canonsphere
cognition     routing     visualization
   │
   ▼
optical-body  (physical apertures)
```

- **MetaField** owns the canonical cone transform.
- **Aurora** routes cones as work units. It does not own the operator. Regions that are not interesting are not expanded.
- **Canonsphere** witnesses. Sphere → stereograph → nested caps. `js/` is the adapter surface.
- **optical-body-s3** supplies observations: each dodecahedral face is a cone aperture. Hardware produces geometry; the transform collapses it.

Neutral protocol: `cone-state-v0.1` (`protocol.py`). Geometry never depends on a backend.

## Tests

```bash
python -m pytest tests/test_cones.py -q
```

## Status

experimental. Geometry is the contract. Neural weights are a backend.
