# telescoping-cones

Geometric attention operator.

Information enters through a wide cross-section, is progressively compressed by nested cones, and exits as a narrow state. Geometry is independent of the neural implementation.

**Live instrument:** [thebabeldragon.github.io/telescoping-cones](https://thebabeldragon.github.io/telescoping-cones/)

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

The operator is not a renamed Transformer layer. Geometry actually modifies attention.

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

## Instrument

Vanilla `pages/` lab, same deploy path as Canonsphere.

- Drag to orbit. Shift-drag or AIM to point the cone.
- SPHERE → STEREOGRAPH morph. A cap of angular radius α becomes a circle of tan(α/2).
- ATTENTION: click a point. Ribbons are the A^cone row.
- AURORA only splits regions that deserve compute.
- MEMORY stores the geometric narrowing, not just the conclusion.
- OPTICAL: dodecahedral apertures, the optical-body witness.

```
pages/                 GitHub Pages instrument
  lab.js               Three.js witness
  js/index.js          same mathematics as telescoping_cones/
.github/workflows/pages.yml
```

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
pages/             live instrument
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
- **Aurora** routes cones as work units. It does not own the operator.
- **Canonsphere** witnesses. Sphere → stereograph → nested caps.
- **optical-body-s3** supplies observations: each dodecahedral face is a cone aperture.

Neutral protocol: `cone-state-v0.1`. Geometry never depends on a backend.

## Tests

```bash
python -m pytest tests/test_cones.py -q
```

## Status

experimental. Geometry is the contract. Neural weights are a backend.
