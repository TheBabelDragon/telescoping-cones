/** Telescoping cones — geometry-first attention. Browser / Canonsphere build. */

export function normalize(a) {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}

export function geodesic(a, b) {
  const A = normalize(a), B = normalize(b);
  const d = Math.min(1, Math.max(-1, A[0] * B[0] + A[1] * B[1] + A[2] * B[2]));
  return Math.acos(d);
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

export function stereographic(points, axis) {
  const a = normalize(axis);
  const c = a[2];
  const rot = (p) => {
    if (c > 0.999999) return p;
    if (c < -0.999999) return [p[0], p[1], -p[2]];
    const v = [-a[1], a[0], 0];
    const k = 1 / (1 + c);
    const vx = [v[1] * p[2] - v[2] * p[1], v[2] * p[0] - v[0] * p[2], v[0] * p[1] - v[1] * p[0]];
    const vxx = [v[1] * vx[2] - v[2] * vx[1], v[2] * vx[0] - v[0] * vx[2], v[0] * vx[1] - v[1] * vx[0]];
    return [p[0] + vx[0] + vxx[0] * k, p[1] + vx[1] + vxx[1] * k, p[2] + vx[2] + vxx[2] * k];
  };
  return points.map((p) => {
    const q = rot(p);
    const d = 1 + q[2];
    const den = Math.abs(d) < 1e-9 ? 1e-9 : d;
    return [q[0] / den, q[1] / den];
  });
}

export function capRadiusOnPlane(aperture) {
  return Math.tan(aperture / 2);
}

export class TelescopingCone {
  constructor({ center, radius, aperture, temperature = 1, apex = [0, 0, 0], metric = "angular", sigma }) {
    this.center = normalize(center);
    this.radius = radius;
    this.aperture = aperture;
    this.temperature = temperature;
    this.apex = apex;
    this.metric = metric;
    this.sigma = sigma ?? Math.max(radius * 0.55, 1e-4);
  }
  angleTo(p) {
    return geodesic([p[0] - this.apex[0], p[1] - this.apex[1], p[2] - this.apex[2]], this.center);
  }
  distance(p) {
    if (this.metric === "angular") return this.angleTo(p);
    return Math.hypot(p[0] - this.apex[0], p[1] - this.apex[1], p[2] - this.apex[2]);
  }
  contains(p) {
    return this.distance(p) < this.radius && this.angleTo(p) < this.aperture;
  }
  weight(p) {
    const T = Math.max(this.temperature, 1e-4);
    const d = this.distance(p);
    const th = this.angleTo(p);
    const radial = Math.exp(-(d * d) / (2 * this.sigma * this.sigma * T));
    const angular = Math.exp(-(th * th) / (2 * this.aperture * this.aperture));
    return radial * angular;
  }
  weights(points) {
    return points.map((p) => this.weight(p));
  }
  collapse(features, points) {
    const w = this.weights(points);
    const mass = w.reduce((a, b) => a + b, 0);
    const dim = features[0]?.length ?? 0;
    if (mass < 1e-12) return { state: Array(dim).fill(0), entropy: 0, mass: 0, count: 0, weights: w };
    const state = Array(dim).fill(0);
    let entropy = 0, count = 0;
    for (let i = 0; i < w.length; i++) {
      const p = w[i] / mass;
      if (w[i] > 1e-4) count += 1;
      if (p > 1e-15) entropy -= p * Math.log(p);
      for (let d = 0; d < dim; d++) state[d] += p * features[i][d];
    }
    return { state, entropy, mass, count, weights: w };
  }
  transform(observations, geometry) {
    return this.collapse(observations, geometry);
  }
}

export const PROTOCOL_VERSION = "cone-state-v0.1";
