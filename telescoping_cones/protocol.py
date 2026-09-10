from __future__ import annotations

import json

import numpy as np

PROTOCOL_VERSION = "cone-state-v0.1"


def fnv1a_hex(s: str) -> str:
    h = 0x811C9DC5
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def _num(x) -> float:
    return float(np.round(float(x), 6))


def encode_cone_state(kind, axis, hierarchy, memory, workers, tick: int) -> dict:
    layers = hierarchy.layers
    last = layers[-1] if layers else None
    payload = {
        "version": PROTOCOL_VERSION,
        "source": {"kind": kind},
        "axis": [_num(x) for x in np.asarray(axis).tolist()],
        "radii": [_num(l.radius) for l in layers],
        "apertures": [_num(l.aperture) for l in layers],
        "collapsed": [_num(x) for x in (last.state.tolist() if last is not None else [])],
        "entropy": [_num(l.entropy) for l in layers],
        "mass": [_num(l.mass) for l in layers],
        "counts": [int(l.count) for l in layers],
        "memory": [_num(x) for x in np.asarray(memory).tolist()],
        "workers": [
            {"id": w.id, "mass": _num(w.mass), "entropy": _num(w.entropy), "expanded": bool(w.expanded)}
            for w in workers
        ],
        "tick": int(tick),
    }
    payload["hash"] = fnv1a_hex(json.dumps(payload, separators=(",", ":")))
    return payload
