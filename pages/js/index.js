/** Telescoping cones — geometry-first attention. Browser / Canonsphere build.
 *  Same mathematics as telescoping_cones/*.py. Geometry never depends on a backend.
 */

export const PROTOCOL_VERSION = "cone-state-v0.1";
export const FEATURE_DIM = 8;
export const LAYER_STEMS = ["perception", "structure", "relation", "salience", "semantic"];
const PHI = (1 + Math.sqrt(5)) / 2;

export function v3(x = 0, y = 0, z = 0) { return [x, y, z]; }
export function clone(a) { return [a[0], a[1], a[2]]; }
export function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function length(a) { return Math.hypot(a[0], a[1], a[2]); }
export function dist(a, b) { return length(sub(a, b)); }
export function normalize(a, eps = 1e-12) {
  const n = length(a);
  if (n < eps) return [0, 0, 1];
  return scale(a, 1 / n);
}
export function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }
export function lerp(a, b, t) { return a + (b - a) * t; }
export function lerp3(a, b, t) { return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]; }
export function angleBetween(a, b) { return Math.acos(clamp(dot(normalize(a), normalize(b)), -1, 1)); }
export function geodesic(a, b) { return angleBetween(a, b); }
export function sigmoid(x) {
  if (x > 20) return 1;
  if (x < -20) return 0;
  return 1 / (1 + Math.exp(-x));
}
export function mulberry32(seed) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function fibonacciSphere(n) {
  const golden = (1 + Math.sqrt(5)) / 2;
  const out = [];
  for (let i = 0; i < n; i++) {
    const z = 1 - (2 * (i + 0.5)) / n;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const theta = (2 * Math.PI * i) / golden;
    out.push([r * Math.cos(theta), r * Math.sin(theta), z]);
  }
  return out;
}

export function tangentBasis(axis) {
  const n = normalize(axis);
  const helper = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const t1 = normalize(cross(helper, n));
  return [t1, cross(n, t1)];
}

/** Rodrigues rotation mapping `axis` onto +Z. */
export function rotateToPole(points, axis) {
  const a = normalize(axis);
  const c = a[2];
  if (c > 0.999999) return points.map((p) => [p[0], p[1], p[2]]);
  if (c < -0.999999) return points.map((p) => [p[0], p[1], -p[2]]);
  const v = [-a[1], a[0], 0];
  const k = 1 / (1 + c);
  return points.map((p) => {
    const vx = cross(v, p);
    const vxx = cross(v, vx);
    return add(add(p, vx), scale(vxx, k));
  });
}

export function rotateFromPole(points, axis) {
  const a = normalize(axis);
  const c = a[2];
  if (c > 0.999999) return points.map((p) => [p[0], p[1], p[2]]);
  if (c < -0.999999) return points.map((p) => [p[0], p[1], -p[2]]);
  const v = [a[1], -a[0], 0];
  const k = 1 / (1 + c);
  return points.map((p) => {
    const vx = cross(v, p);
    const vxx = cross(v, vx);
    return add(add(p, vx), scale(vxx, k));
  });
}

/**
 * Stereographic projection from the south pole after rotating `axis` to +Z.
 * A cap of angular radius α maps to a circle of tan(α/2).
 */
export function stereographic(points, axis) {
  return rotateToPole(points, axis).map(([x, y, z]) => {
    const den = Math.abs(1 + z) < 1e-9 ? 1e-9 : 1 + z;
    return [x / den, y / den];
  });
}

export function capRadiusOnPlane(aperture) {
  return Math.tan(aperture / 2);
}

/** Face centers of a regular dodecahedron = icosahedron vertices. */
export function dodecahedronFaces() {
  const raw = [];
  for (const s1 of [-1, 1]) {
    for (const s2 of [-1, 1]) {
      raw.push([0, s1, s2 * PHI]);
      raw.push([s1, s2 * PHI, 0]);
      raw.push([s1 * PHI, 0, s2]);
    }
  }
  return raw.map((p) => normalize(p));
}

export function angularGate(theta, aperture) {
  const a = Math.max(aperture, 1e-6);
  return Math.exp(-(theta * theta) / (2 * a * a));
}

export function radialGate(distance, sigma, temperature = 1) {
  const s = Math.max(sigma, 1e-6);
  const t = Math.max(temperature, 1e-4);
  return Math.exp(-(distance * distance) / (2 * s * s * t));
}

export function offsetAxis(axis, tangent, angle) {
  return normalize(add(scale(normalize(axis), Math.cos(angle)), scale(normalize(tangent), Math.sin(angle))));
}

export function principalTangent(axis, points, weights) {
  const [t1, t2] = tangentBasis(axis);
  let s11 = 0, s12 = 0, s22 = 0;
  for (let i = 0; i < points.length; i++) {
    const w = weights[i] ?? 0;
    if (w <= 1e-8) continue;
    const u = dot(points[i], t1);
    const v = dot(points[i], t2);
    s11 += w * u * u;
    s12 += w * u * v;
    s22 += w * v * v;
  }
  const diff = s11 - s22;
  const angle = 0.5 * Math.atan2(2 * s12, diff === 0 && s12 === 0 ? 1 : diff);
  return normalize(add(scale(t1, Math.cos(angle)), scale(t2, Math.sin(angle))));
}

export class TelescopingCone {
  constructor({ center, radius, aperture, temperature = 1, apex = [0, 0, 0], metric = "angular", sigma, depth = 1 }) {
    this.center = normalize(center);
    this.radius = radius;
    this.aperture = aperture;
    this.temperature = temperature;
    this.apex = apex;
    this.metric = metric;
    this.depth = depth;
    this.sigma = sigma ?? Math.max(radius * 0.55, 1e-4);
  }
  angleTo(p) {
    return geodesic(sub(p, this.apex), this.center);
  }
  distance(p) {
    if (this.metric === "angular") return this.angleTo(p);
    return length(sub(p, this.apex));
  }
  contains(p) {
    return this.distance(p) < this.radius && this.angleTo(p) < this.aperture;
  }
  weight(p) {
    return radialGate(this.distance(p), this.sigma, this.temperature) * angularGate(this.angleTo(p), this.aperture);
  }
  weights(points) {
    const w = new Float64Array(points.length);
    for (let i = 0; i < points.length; i++) w[i] = this.weight(points[i]);
    return w;
  }
  collapse(features, points) {
    const w = this.weights(points);
    let mass = 0;
    for (let i = 0; i < w.length; i++) mass += w[i];
    const dim = features[0]?.length ?? 0;
    if (mass < 1e-12 || dim === 0) {
      return { state: Array(dim).fill(0), entropy: 0, mass: 0, count: 0, weights: w };
    }
    const state = Array(dim).fill(0);
    let entropy = 0, count = 0;
    for (let i = 0; i < w.length; i++) {
      const p = w[i] / mass;
      if (w[i] > 1e-4) count += 1;
      if (p > 1e-15) entropy -= p * Math.log(p);
      const f = features[i];
      if (!f) continue;
      for (let d = 0; d < dim; d++) state[d] += p * (f[d] ?? 0);
    }
    return { state, entropy, mass, count, weights: w };
  }
  transform(observations, geometry) {
    return this.collapse(observations, geometry);
  }
}

export function layerName(k, depth) {
  if (k === depth - 1) return "state";
  return LAYER_STEMS[k] ?? `cone-${k}`;
}

/** r_{k+1} = r_k · σ(f(X_k)). Uncertainty keeps the next cone open. */
export function nextShrink(entropy, count, adaptive) {
  if (!adaptive) return 0.58;
  const Hmax = Math.log(Math.max(count, 2));
  const Hnorm = clamp(entropy / Hmax, 0, 1);
  const open = sigmoid(Hnorm * 6 - 2.4);
  return clamp(0.38 + 0.5 * open, 0.34, 0.88);
}

export function buildHierarchy({ axis, aperture0, radius0, depth, temperature, adaptive, metric = "angular", points, features, apex }) {
  const D = clamp(Math.round(depth), 2, 6);
  const layers = [];
  let aperture = aperture0;
  let radius = radius0;
  for (let k = 0; k < D; k++) {
    const cone = new TelescopingCone({ center: axis, radius, depth: k, aperture, temperature, apex, metric });
    const collapsed = cone.collapse(features, points);
    const shrink = k === D - 1 ? 1 : nextShrink(collapsed.entropy, collapsed.count, adaptive);
    layers.push({ k, name: layerName(k, D), aperture, radius, shrink, cone, ...collapsed });
    aperture *= shrink;
    radius *= shrink;
  }
  return { layers, axis };
}

export function coneAffinity(points, apex, sigma, aperture) {
  const n = points.length;
  const A = new Float64Array(n * n);
  const dirs = points.map((p) => normalize(sub(p, apex)));
  const s2 = 2 * Math.max(sigma, 1e-6) ** 2;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const d = dist(points[i], points[j]);
      const theta = Math.acos(clamp(dot(dirs[i], dirs[j]), -1, 1));
      A[i * n + j] = Math.exp(-(d * d) / s2) * angularGate(theta, aperture);
    }
  }
  return A;
}

export function makeProjection(dim, seed) {
  const rng = mulberry32(seed);
  const rows = [];
  for (let i = 0; i < dim; i++) {
    const row = Array.from({ length: dim }, () => rng() * 2 - 1);
    for (let k = 0; k < i; k++) {
      let acc = 0;
      for (let d = 0; d < dim; d++) acc += row[d] * rows[k][d];
      for (let d = 0; d < dim; d++) row[d] -= acc * rows[k][d];
    }
    let n = 0;
    for (let d = 0; d < dim; d++) n += row[d] * row[d];
    n = Math.sqrt(Math.max(n, 1e-12));
    rows.push(row.map((x) => x / n));
  }
  return rows;
}

function matVec(M, v) {
  return M.map((row) => row.reduce((acc, x, i) => acc + x * (v[i] ?? 0), 0));
}

function softmaxRow(logits, offset, n) {
  let max = -Infinity;
  for (let j = 0; j < n; j++) max = Math.max(max, logits[offset + j]);
  let sum = 0;
  for (let j = 0; j < n; j++) {
    const e = Math.exp(logits[offset + j] - max);
    logits[offset + j] = e;
    sum += e;
  }
  const inv = 1 / Math.max(sum, 1e-12);
  for (let j = 0; j < n; j++) logits[offset + j] *= inv;
}

/** Attention(Q,K,V) = softmax(QKᵀ/√d + λ A^cone) V */
export function coneAttention(features, affinity, lambda, temperature, Wq, Wk, Wv) {
  const n = features.length;
  const dim = features[0]?.length ?? 0;
  const Q = features.map((f) => matVec(Wq, f));
  const K = features.map((f) => matVec(Wk, f));
  const V = features.map((f) => matVec(Wv, f));
  const scaleQ = 1 / Math.sqrt(Math.max(dim, 1));
  const T = Math.max(temperature, 1e-4);
  const weights = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let d = 0; d < dim; d++) s += Q[i][d] * K[j][d];
      weights[i * n + j] = (s * scaleQ + lambda * affinity[i * n + j]) / T;
    }
    softmaxRow(weights, i * n, n);
  }
  const output = [];
  for (let i = 0; i < n; i++) {
    const row = Array(dim).fill(0);
    for (let j = 0; j < n; j++) {
      const a = weights[i * n + j];
      const v = V[j];
      for (let d = 0; d < dim; d++) row[d] += a * v[d];
    }
    output.push(row);
  }
  return { output, weights };
}

export function defaultProjections(dim) {
  return { Wq: makeProjection(dim, 1112), Wk: makeProjection(dim, 2213), Wv: makeProjection(dim, 3324) };
}

const ENTROPY_EXPAND = 1.15;
const MIN_MASS = 0.35;

function walkWorker(id, axis, aperture, radius, points, features, depthLeft, temperature) {
  const cone = new TelescopingCone({ center: axis, radius, aperture, temperature });
  const c = cone.collapse(features, points);
  const shouldExpand = depthLeft > 0 && c.entropy > ENTROPY_EXPAND && c.mass > MIN_MASS && c.count > 6;
  const node = { id, axis, aperture, radius, mass: c.mass, entropy: c.entropy, count: c.count, expanded: shouldExpand, children: [] };
  if (!shouldExpand) return node;
  const tangent = principalTangent(axis, points, c.weights);
  const split = Math.min(aperture * 0.55, 0.32);
  const shrink = nextShrink(c.entropy, c.count, true);
  const childAperture = aperture * shrink;
  const childRadius = radius * shrink;
  node.children = [
    walkWorker(`${id}1`, offsetAxis(axis, tangent, split), childAperture, childRadius, points, features, depthLeft - 1, temperature),
    walkWorker(`${id}2`, offsetAxis(axis, tangent, -split), childAperture, childRadius, points, features, depthLeft - 1, temperature),
  ];
  return node;
}

function flatten(node, acc) {
  acc.push(node);
  for (const child of node.children) flatten(child, acc);
}

/** Aurora owns routing, not the cone transform. */
export function routeCones(axis, aperture, radius, points, features, temperature = 1, maxDepth = 2) {
  const root = walkWorker("A", axis, aperture, radius, points, features, maxDepth, temperature);
  const flat = [];
  flatten(root, flat);
  return { root, flat };
}

export class ConeMemory {
  constructor(dim, alpha = 0.86, maxTraces = 12) {
    this.alpha = alpha;
    this.state = Array(dim).fill(0);
    this.traces = [];
    this.maxTraces = maxTraces;
    this.tick = 0;
  }
  update(collapsed, axis, radius, aperture, alpha = this.alpha) {
    if (this.state.length !== collapsed.length) this.state = collapsed.slice();
    else {
      for (let i = 0; i < collapsed.length; i++) this.state[i] = alpha * this.state[i] + (1 - alpha) * collapsed[i];
    }
    this.tick += 1;
    this.traces.unshift({ axis: clone(axis), radius, aperture, state: collapsed.slice(), t: this.tick });
    if (this.traces.length > this.maxTraces) this.traces.pop();
  }
}

function featureFromClusters(p, clusters, rng) {
  const f = [];
  for (let d = 0; d < FEATURE_DIM; d++) {
    const c = clusters[d % clusters.length];
    f.push(Math.exp(-(dist(p, c) ** 2) / 0.55) + (rng() - 0.5) * 0.08);
  }
  return f;
}

export function generateSphereField(n = 96, seed = 1112) {
  const rng = mulberry32(seed);
  const pts = fibonacciSphere(n);
  const clusters = [[0.72, 0.48, 0.5], [-0.2, 0.86, -0.46], [-0.64, -0.42, 0.64]];
  return pts.map((p, id) => ({ id, p, feature: featureFromClusters(p, clusters, rng), source: "sphere" }));
}

export function generateOpticalField(seed = 1112) {
  const rng = mulberry32(seed);
  const faces = dodecahedronFaces();
  const laserOn = faces.map((_, i) => i % 2 === 0 && rng() > 0.25);
  return faces.map((n, id) => {
    let intensity = 0;
    for (let k = 0; k < faces.length; k++) {
      if (laserOn[k]) intensity += Math.max(0, dot(n, faces[k])) ** 3;
    }
    intensity = clamp(intensity, 0, 2);
    const feature = Array.from({ length: FEATURE_DIM }, (_, d) => {
      if (d === 0) return intensity;
      if (d === 1) return intensity * intensity;
      return intensity * Math.max(0, dot(n, faces[(id + d) % faces.length]));
    });
    return { id, p: n, feature, source: "photodiode", face: id, laser: laserOn[id] };
  });
}

export function generateLatticeField(n = 4, seed = 1112) {
  const rng = mulberry32(seed);
  const out = [];
  let id = 0;
  const span = 0.85;
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) {
      for (let z = 0; z < n; z++) {
        const p = [
          -span + (2 * span * x) / (n - 1),
          -span + (2 * span * y) / (n - 1),
          -span + (2 * span * z) / (n - 1),
        ];
        out.push({
          id,
          p,
          feature: [
            Math.sin(p[0] * 3.1), Math.cos(p[1] * 2.7), Math.sin(p[2] * 3.7),
            Math.cos((p[0] + p[1]) * 1.9), Math.sin((p[1] * p[2]) * 4.2),
            p[0] * p[1], p[1] * p[2], rng() * 0.2,
          ],
          source: "lattice",
        });
        id += 1;
      }
    }
  }
  return out;
}

export function generateField(kind, seed = 1112) {
  if (kind === "optical") return generateOpticalField(seed);
  if (kind === "lattice") return generateLatticeField(4, seed);
  return generateSphereField(96, seed);
}

export function driftFeatures(obs, dt, uncertainty, rng) {
  const amp = uncertainty * 0.35 * dt;
  for (const o of obs) {
    for (let d = 0; d < o.feature.length; d++) {
      o.feature[d] = clamp(o.feature[d] + (rng() - 0.5) * amp, -2, 2);
    }
    if (o.source === "sphere") {
      const q = add(o.p, scale(v3(rng() - 0.5, rng() - 0.5, rng() - 0.5), dt * 0.02 * uncertainty));
      o.p = normalize(q);
    }
  }
}

export function fnv1aHex(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

export function encodeConeState({ kind, axis, hierarchy, memory, workers, tick }) {
  const layers = hierarchy.layers;
  const last = layers[layers.length - 1];
  const payload = {
    version: PROTOCOL_VERSION,
    source: { kind },
    axis: axis.map((x) => Number(x.toFixed(6))),
    radii: layers.map((l) => Number(l.radius.toFixed(6))),
    apertures: layers.map((l) => Number(l.aperture.toFixed(6))),
    collapsed: (last?.state ?? []).map((x) => Number(x.toFixed(6))),
    entropy: layers.map((l) => Number(l.entropy.toFixed(6))),
    mass: layers.map((l) => Number(l.mass.toFixed(6))),
    counts: layers.map((l) => l.count),
    memory: memory.map((x) => Number(x.toFixed(6))),
    workers: workers.map((w) => ({
      id: w.id,
      mass: Number(w.mass.toFixed(4)),
      entropy: Number(w.entropy.toFixed(4)),
      expanded: w.expanded,
    })),
    tick,
  };
  return { ...payload, hash: fnv1aHex(JSON.stringify(payload)) };
}
