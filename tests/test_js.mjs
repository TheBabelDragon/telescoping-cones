import assert from "node:assert/strict";
import { capRadiusOnPlane, fibonacciSphere, stereographic, TelescopingCone } from "../js/index.js";

const points = fibonacciSphere(32);
const cone = new TelescopingCone({ center: [0, 0, 1], radius: 0.8, aperture: 0.7 });
const w = cone.weights(points);
assert.equal(w.length, 32);
assert.ok(w.some((x) => x > 0.1));

const alpha = 0.5;
const t = [...Array(12)].map((_, i) => (i / 12) * Math.PI * 2);
const ring = t.map((a) => [Math.sin(alpha) * Math.cos(a), Math.sin(alpha) * Math.sin(a), Math.cos(alpha)]);
const uv = stereographic(ring, [0, 0, 1]);
const expected = capRadiusOnPlane(alpha);
for (const [x, y] of uv) {
  assert.ok(Math.abs(Math.hypot(x, y) - expected) < 1e-6);
}

console.log("js ok");
