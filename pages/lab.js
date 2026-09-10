/**
 * Telescoping Cones instrument.
 * Geometry is the operator. This file only witnesses it.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  TelescopingCone,
  buildHierarchy,
  coneAffinity,
  coneAttention,
  defaultProjections,
  routeCones,
  ConeMemory,
  generateField,
  driftFeatures,
  encodeConeState,
  stereographic,
  capRadiusOnPlane,
  rotateToPole,
  rotateFromPole,
  dodecahedronFaces,
  mulberry32,
  normalize,
  lerp,
  clamp,
} from "./js/index.js";

const ICE = 0x9fd8e8;
const PAPER = 0xececf2;
const DIM = 0x3a424c;
const BG = 0x07070a;

const tmp = new THREE.Vector3();
const tmp2 = new THREE.Vector3();
const tmpQ = new THREE.Quaternion();
const tmpM = new THREE.Matrix4();
const yAxis = new THREE.Vector3(0, 1, 0);
const color = new THREE.Color();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function alignY(axis, target) {
  tmp.copy(axis).normalize();
  target.setFromUnitVectors(yAxis, tmp);
}

function bits(h) {
  return h / Math.LN2;
}

const $ = (id) => document.getElementById(id);

const state = {
  mode: "field",
  fieldKind: "sphere",
  playing: true,
  aiming: false,
  adaptive: true,
  aperture: 0.38,
  depth: 4,
  lambda: 0.8,
  temperature: 1,
  morph: 0,
  targetMorph: 0,
  uncertainty: 0.45,
  axis: [0.42, 0.58, 0.69],
  displayAxis: new THREE.Vector3(0.42, 0.58, 0.69).normalize(),
  selected: 0,
  tick: 0,
  seed: 1112,
  obs: [],
  hierarchy: { layers: [] },
  affinity: null,
  attention: null,
  routing: { flat: [] },
  memory: new ConeMemory(8),
  protocol: null,
  proj: defaultProjections(8),
  rng: mulberry32(1112),
  pulse: 0,
};

function setField(kind, seed = state.seed) {
  state.fieldKind = kind;
  state.obs = generateField(kind, seed);
  state.memory = new ConeMemory(state.obs[0]?.feature.length ?? 8);
  state.selected = 0;
  recompute(true);
}

function recompute(forceAttention = false) {
  const points = state.obs.map((o) => o.p);
  const features = state.obs.map((o) => o.feature);
  state.hierarchy = buildHierarchy({
    axis: state.axis,
    aperture0: state.aperture,
    radius0: 1.05,
    depth: state.depth,
    temperature: state.temperature,
    adaptive: state.adaptive,
    points,
    features,
  });
  const last = state.hierarchy.layers.at(-1);
  if (last) state.memory.update(last.state, state.axis, last.radius, last.aperture);
  state.routing = routeCones(state.axis, state.aperture, 1.05, points, features, state.temperature, 2);
  if (forceAttention || state.mode === "attention") {
    const sigma = Math.max(state.aperture * 0.7, 0.08);
    state.affinity = coneAffinity(points, [0, 0, 0], sigma, state.aperture);
    state.attention = coneAttention(features, state.affinity, state.lambda, state.temperature, state.proj.Wq, state.proj.Wk, state.proj.Wv);
  }
  state.tick += 1;
  state.protocol = encodeConeState({
    kind: state.fieldKind,
    axis: state.axis,
    hierarchy: state.hierarchy,
    memory: state.memory.state,
    workers: state.routing.flat,
    tick: state.tick,
  });
  paintHud();
}

function paintHud() {
  const layers = state.hierarchy.layers;
  const first = layers[0];
  const last = layers.at(-1);
  const list = $("layer-list");
  if (list) {
    list.innerHTML = layers
      .map((layer) => {
        const w = first ? (layer.count / Math.max(first.count, 1)) * 100 : 0;
        return `<li><span>${layer.name.toUpperCase()}</span><span class="bar"><i style="width:${w}%"></i></span><span class="n">n=${layer.count}</span></li>`;
      })
      .join("");
  }
  if ($("layer-meta") && last && first) {
    const compression = first.count / Math.max(last.count, 1);
    $("layer-meta").innerHTML = `<span>H ${bits(last.entropy).toFixed(2)} bit</span><span>×${compression.toFixed(1)} compress</span>`;
    const Hnorm = Math.min(1, last.entropy / Math.log(Math.max(last.count, 2)));
    $("explore-dot").style.left = `${(1 - Hnorm) * 100}%`;
  }
  const p = state.protocol;
  if ($("card") && p) {
    $("card").textContent =
      `HASH   ${p.hash}\nTICK   ${String(p.tick).padStart(4, "0")}\nAXIS   ${p.axis.map((n) => n.toFixed(3)).join(" ")}\nDEPTH  ${p.radii.length}\nSTATE  ${p.collapsed.map((n) => n.toFixed(2)).join(" ")}\nWORK   ${p.workers.length}`;
  }
  if ($("status") && p) {
    $("status").textContent = `${p.hash} · ${state.fieldKind} · λ ${state.lambda.toFixed(2)}`;
    $("status").className = "ok";
  }
}

class Instrument {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    this.renderer.setClearColor(BG, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(BG, 5.2, 13);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 40);
    this.camera.position.set(0.2, 0.55, 3.4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1.7;
    this.controls.maxDistance = 6.8;
    this.controls.rotateSpeed = 0.7;

    this.scene.add(new THREE.HemisphereLight(0xb8c4d4, 0x0a0a10, 0.55));
    const key = new THREE.DirectionalLight(0xe8f4ff, 1.1);
    key.position.set(2.1, 3.1, 4);
    this.scene.add(key);
    this.scene.add(new THREE.AmbientLight(ICE, 0.08));

    this.sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.985, 64, 48),
      new THREE.MeshStandardMaterial({
        color: 0x14141c,
        metalness: 0.18,
        roughness: 0.52,
        transparent: true,
        opacity: 0.42,
        depthWrite: false,
      }),
    );
    this.scene.add(this.sphere);

    this.wire = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.002, 2),
      new THREE.MeshBasicMaterial({ color: ICE, wireframe: true, transparent: true, opacity: 0.22 }),
    );
    this.scene.add(this.wire);

    this.addMeridianGrid();

    this.handle = new THREE.Mesh(
      new THREE.SphereGeometry(0.05, 20, 16),
      new THREE.MeshStandardMaterial({ color: PAPER, emissive: ICE, emissiveIntensity: 0.45, roughness: 0.28 }),
    );
    this.scene.add(this.handle);

    this.lance = new THREE.Mesh(
      new THREE.CylinderGeometry(0.008, 0.008, 2.3, 8),
      new THREE.MeshBasicMaterial({ color: PAPER, transparent: true, opacity: 0.55 }),
    );
    this.scene.add(this.lance);

    this.points = new THREE.InstancedMesh(
      new THREE.IcosahedronGeometry(1, 0),
      new THREE.MeshBasicMaterial(),
      128,
    );
    this.points.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    const coneGeo = new THREE.ConeGeometry(1, 1, 48, 1, true);
    coneGeo.translate(0, -0.5, 0);
    this.coneFill = new THREE.Mesh(
      coneGeo,
      new THREE.MeshBasicMaterial({
        color: ICE,
        transparent: true,
        opacity: 0.05,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.scene.add(this.coneFill);

    this.rings = [];
    for (let i = 0; i < 6; i++) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.985, 1.018, 80),
        new THREE.MeshBasicMaterial({
          color: ICE,
          transparent: true,
          opacity: 0.22 + i * 0.1,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
      ring.visible = false;
      this.rings.push(ring);
      this.scene.add(ring);
    }

    this.pulseRing = new THREE.Mesh(
      new THREE.RingGeometry(0.97, 1.04, 80),
      new THREE.MeshBasicMaterial({
        color: PAPER,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.scene.add(this.pulseRing);

    this.flow = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    this.flow.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(48 * 2 * 3), 3));
    this.flow.geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(48 * 2 * 3), 3));
    this.scene.add(this.flow);

    this.ghosts = [];
    for (let i = 0; i < 10; i++) {
      const g = new THREE.Mesh(
        new THREE.RingGeometry(0.988, 1.012, 48),
        new THREE.MeshBasicMaterial({ color: ICE, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }),
      );
      g.visible = false;
      this.ghosts.push(g);
      this.scene.add(g);
    }

    this.aurora = [];
    for (let i = 0; i < 8; i++) {
      const a = new THREE.Mesh(
        new THREE.RingGeometry(0.986, 1.02, 56),
        new THREE.MeshBasicMaterial({ color: PAPER, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }),
      );
      a.visible = false;
      this.aurora.push(a);
      this.scene.add(a);
    }

    const faces = dodecahedronFaces();
    const dPos = [];
    // pentagon-ish wire: connect each face to its 3 nearest neighbors
    for (let i = 0; i < faces.length; i++) {
      const scored = faces
        .map((f, j) => ({ j, d: f[0] * faces[i][0] + f[1] * faces[i][1] + f[2] * faces[i][2] }))
        .filter((x) => x.j !== i)
        .sort((a, b) => b.d - a.d)
        .slice(0, 3);
      for (const n of scored) {
        if (n.j > i) {
          dPos.push(...faces[i], ...faces[n.j]);
        }
      }
    }
    this.dodeca = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: ICE, transparent: true, opacity: 0.55 }),
    );
    this.dodeca.geometry.setAttribute("position", new THREE.Float32BufferAttribute(dPos, 3));
    this.dodeca.visible = false;
    this.scene.add(this.dodeca);

    this.lasers = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xffc86a, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending }),
    );
    this.lasers.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(12 * 2 * 3), 3));
    this.lasers.visible = false;
    this.scene.add(this.lasers);

    this.stereoGroup = new THREE.Group();
    this.scene.add(this.stereoGroup);
    this.stereoPlane = new THREE.Mesh(
      new THREE.CircleGeometry(1.65, 72),
      new THREE.MeshBasicMaterial({ color: 0x10141a, transparent: true, opacity: 0.0, side: THREE.DoubleSide, depthWrite: false }),
    );
    this.stereoGroup.add(this.stereoPlane);
    this.stereoRings = [];
    for (let i = 0; i < 6; i++) {
      const g = new THREE.RingGeometry(0.98, 1.02, 80);
      const m = new THREE.Mesh(
        g,
        new THREE.MeshBasicMaterial({ color: ICE, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false }),
      );
      m.rotation.x = -Math.PI / 2;
      m.visible = false;
      this.stereoRings.push(m);
      this.stereoGroup.add(m);
    }

    this.pickPlane = new THREE.Mesh(
      new THREE.SphereGeometry(1.02, 32, 24),
      new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide }),
    );
    this.scene.add(this.pickPlane);

    this.clock = new THREE.Clock();
    this.accum = 0;
    this.draggingAim = false;
    this.bind();
    this.resize();
    this.renderer.render(this.scene, this.camera);
    this.loop = this.loop.bind(this);
    this.renderer.setAnimationLoop(this.loop);
  }

  addMeridianGrid() {
    const pos = [];
    const rings = 6;
    const segs = 64;
    for (let r = 1; r <= rings; r++) {
      const th = (r / (rings + 1)) * Math.PI;
      const rad = Math.sin(th);
      const y = Math.cos(th);
      for (let i = 0; i < segs; i++) {
        const a0 = (i / segs) * Math.PI * 2;
        const a1 = ((i + 1) / segs) * Math.PI * 2;
        pos.push(rad * Math.cos(a0), y, rad * Math.sin(a0), rad * Math.cos(a1), y, rad * Math.sin(a1));
      }
    }
    for (let m = 0; m < 8; m++) {
      const phi = (m / 8) * Math.PI;
      for (let i = 0; i < segs; i++) {
        const t0 = (i / segs) * Math.PI * 2;
        const t1 = ((i + 1) / segs) * Math.PI * 2;
        pos.push(Math.sin(t0) * Math.cos(phi), Math.cos(t0), Math.sin(t0) * Math.sin(phi));
        pos.push(Math.sin(t1) * Math.cos(phi), Math.cos(t1), Math.sin(t1) * Math.sin(phi));
      }
    }
    this.grid = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: ICE, transparent: true, opacity: 0.12 }),
    );
    this.grid.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    this.scene.add(this.grid);
  }

  bind() {
    window.addEventListener("resize", () => this.resize());
    this.canvas.addEventListener("pointerdown", (e) => this.onDown(e));
    window.addEventListener("pointermove", (e) => this.onMove(e));
    window.addEventListener("pointerup", () => {
      this.draggingAim = false;
      this.controls.enabled = !state.aiming;
    });
    this.canvas.addEventListener("dblclick", (e) => {
      const hit = this.hitSphere(e);
      if (hit) {
        state.axis = normalize([hit.x, hit.y, hit.z]);
        recompute(true);
        state.pulse = 1;
      }
    });
    this.canvas.addEventListener("click", (e) => {
      if (state.mode !== "attention") return;
      const id = this.hitPoint(e);
      if (id >= 0) {
        state.selected = id;
        recompute(true);
      }
    });
  }

  ndc(e) {
    const r = this.canvas.getBoundingClientRect();
    pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    pointer.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  }

  hitSphere(e) {
    this.ndc(e);
    raycaster.setFromCamera(pointer, this.camera);
    const hits = raycaster.intersectObject(this.pickPlane);
    return hits[0]?.point ?? null;
  }

  hitPoint(e) {
    this.ndc(e);
    raycaster.setFromCamera(pointer, this.camera);
    const hits = raycaster.intersectObject(this.points);
    return hits.length ? hits[0].instanceId ?? -1 : -1;
  }

  onDown(e) {
    if (state.aiming || e.shiftKey) {
      this.draggingAim = true;
      this.controls.enabled = false;
      this.aimFromEvent(e);
    }
  }

  onMove(e) {
    if (!this.draggingAim) return;
    this.aimFromEvent(e);
  }

  aimFromEvent(e) {
    const hit = this.hitSphere(e);
    if (!hit) return;
    state.axis = normalize([hit.x, hit.y, hit.z]);
    recompute(false);
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
  }

  morphPoint(p, axis, morph) {
    if (morph < 0.002) return p;
    const pole = rotateToPole([p], axis)[0];
    const den = Math.abs(1 + pole[2]) < 1e-9 ? 1e-9 : 1 + pole[2];
    const stereo = [pole[0] / den, pole[1] / den, 0];
    const k = 0.72;
    const flat = [stereo[0] * k, stereo[1] * k, 0.02];
    const mixed = [lerp(pole[0], flat[0], morph), lerp(pole[1], flat[1], morph), lerp(pole[2], flat[2], morph)];
    return rotateFromPole([mixed], axis)[0];
  }

  loop() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    this.controls.update();

    state.morph += (state.targetMorph - state.morph) * (1 - Math.exp(-dt * 7));
    tmp2.set(state.axis[0], state.axis[1], state.axis[2]).normalize();
    state.displayAxis.lerp(tmp2, 1 - Math.exp(-dt * 10));
    const axis = [state.displayAxis.x, state.displayAxis.y, state.displayAxis.z];

    if (state.playing) {
      this.accum += dt;
      if (this.accum > 0.09) {
        driftFeatures(state.obs, this.accum, state.uncertainty, state.rng);
        recompute(state.mode === "attention");
        this.accum = 0;
      }
    }

    if (state.pulse > 0) state.pulse = Math.max(0, state.pulse - dt * 1.6);

    this.syncScene(axis, dt);
    this.renderer.render(this.scene, this.camera);
  }

  syncScene(axis, dt) {
    const morph = state.morph;
    const layers = state.hierarchy.layers;
    const last = layers.at(-1);
    const weights = last?.weights;
    const optical = state.mode === "optical" || state.fieldKind === "optical";
    const n = state.obs.length;

    this.sphere.visible = morph < 0.92 && !optical;
    this.sphere.material.opacity = lerp(0.42, 0.0, morph);
    this.wire.visible = morph < 0.85 && !optical;
    this.wire.material.opacity = lerp(0.22, 0.0, morph);
    this.grid.visible = morph < 0.8 && !optical;
    this.grid.material.opacity = lerp(0.12, 0.0, morph);
    this.dodeca.visible = optical;
    this.lasers.visible = optical;

    alignY(state.displayAxis, tmpQ);
    this.handle.position.copy(state.displayAxis).multiplyScalar(1.08);
    this.lance.quaternion.copy(tmpQ);
    this.lance.material.opacity = lerp(0.5, 0.15, morph);

    const outer = layers[0];
    const coneH = 1.05;
    const coneR = Math.sin(outer?.aperture ?? state.aperture);
    this.coneFill.visible = morph < 0.75 && !optical;
    this.coneFill.quaternion.copy(tmpQ);
    this.coneFill.position.copy(state.displayAxis).multiplyScalar(coneH * 0.5);
    this.coneFill.scale.set(coneR, coneH, coneR);
    this.coneFill.material.opacity = lerp(0.05, 0, morph) + state.pulse * 0.08;

    for (let i = 0; i < this.rings.length; i++) {
      const layer = layers[i];
      const ring = this.rings[i];
      if (!layer || optical) {
        ring.visible = false;
        continue;
      }
      const a = layer.aperture;
      const r = Math.sin(a);
      const y = Math.cos(a);
      ring.visible = morph < 0.98;
      ring.quaternion.copy(tmpQ);
      const world = this.morphPoint([axis[0] * y, axis[1] * y, axis[2] * y], axis, morph);
      // keep ring on sphere when morph=0, flatten toward plane when morph=1
      ring.position.set(world[0], world[1], world[2]);
      const planeR = capRadiusOnPlane(a) * 0.72;
      const s = lerp(r, planeR, morph);
      ring.scale.set(s, s, s);
      ring.material.opacity = (0.18 + i * 0.1) * lerp(1, 0.85, morph);
    }

    this.pulseRing.visible = state.pulse > 0.02 && !optical;
    if (this.pulseRing.visible) {
      const t = 1 - state.pulse;
      const a = lerp(outer?.aperture ?? 0.4, 0.04, t);
      const r = Math.sin(a);
      this.pulseRing.quaternion.copy(tmpQ);
      this.pulseRing.position.copy(state.displayAxis).multiplyScalar(Math.cos(a));
      this.pulseRing.scale.setScalar(r);
      this.pulseRing.material.opacity = state.pulse * 0.7;
    }

    this.points.count = n;
    const qSel = state.selected;
    const att = state.attention?.weights;
    for (let i = 0; i < n; i++) {
      const o = state.obs[i];
      const mp = this.morphPoint(o.p, axis, morph);
      const w = weights ? weights[i] : 0;
      const attW = att ? att[qSel * n + i] : 0;
      let s = 0.018 + 0.034 * Math.min(1, w * 4);
      if (state.mode === "attention") s = 0.014 + 0.07 * Math.min(1, attW * n * 0.45);
      if (optical) s = 0.055;
      tmpM.compose(tmp.set(mp[0], mp[1], mp[2]), tmpQ.identity(), tmp2.set(s, s, s));
      this.points.setMatrixAt(i, tmpM);
      if (state.mode === "attention") {
        color.setRGB(lerp(0.25, 0.95, attW * n * 0.5), lerp(0.32, 0.9, attW * n * 0.5), lerp(0.38, 0.72, 1 - attW * n * 0.3));
      } else if (optical) {
        const inten = clamp(o.feature[0] / 2, 0, 1);
        color.setRGB(lerp(0.2, 1, inten), lerp(0.35, 0.82, inten), lerp(0.4, 0.45, inten));
      } else {
        const inside = w > 0.04;
        if (inside) color.setRGB(0.72 + 0.25 * w, 0.86, 0.9);
        else color.setRGB(0.28, 0.32, 0.38);
      }
      if (i === qSel && state.mode === "attention") color.setRGB(1, 0.95, 0.82);
      this.points.setColorAt(i, color);
    }
    this.points.instanceMatrix.needsUpdate = true;
    if (this.points.instanceColor) this.points.instanceColor.needsUpdate = true;

    // attention flow ribbons
    const flowPos = this.flow.geometry.attributes.position;
    const flowCol = this.flow.geometry.attributes.color;
    flowPos.array.fill(0);
    flowCol.array.fill(0);
    this.flow.visible = state.mode === "attention" && att && morph < 0.85;
    if (this.flow.visible) {
      const scored = [];
      for (let j = 0; j < n; j++) if (j !== qSel) scored.push([j, att[qSel * n + j]]);
      scored.sort((a, b) => b[1] - a[1]);
      const qp = this.morphPoint(state.obs[qSel].p, axis, morph);
      const top = scored.slice(0, 18);
      top.forEach(([j, w], k) => {
        const tp = this.morphPoint(state.obs[j].p, axis, morph);
        const o = k * 6;
        flowPos.array[o] = qp[0]; flowPos.array[o + 1] = qp[1]; flowPos.array[o + 2] = qp[2];
        flowPos.array[o + 3] = tp[0]; flowPos.array[o + 4] = tp[1]; flowPos.array[o + 5] = tp[2];
        const a = clamp(w * n * 0.55, 0.05, 1);
        flowCol.array[o] = 0.62 * a; flowCol.array[o + 1] = 0.85 * a; flowCol.array[o + 2] = 0.92 * a;
        flowCol.array[o + 3] = 0.35 * a; flowCol.array[o + 4] = 0.55 * a; flowCol.array[o + 5] = 0.7 * a;
      });
      flowPos.needsUpdate = true;
      flowCol.needsUpdate = true;
    }

    const showMem = state.mode === "memory";
    for (let i = 0; i < this.ghosts.length; i++) {
      const g = this.ghosts[i];
      const tr = state.memory.traces[i];
      if (!showMem || !tr) {
        g.visible = false;
        continue;
      }
      const ax = new THREE.Vector3(tr.axis[0], tr.axis[1], tr.axis[2]).normalize();
      alignY(ax, tmpQ);
      const a = tr.aperture;
      g.visible = true;
      g.quaternion.copy(tmpQ);
      g.position.copy(ax).multiplyScalar(Math.cos(a));
      g.scale.setScalar(Math.sin(a));
      g.material.opacity = 0.22 * (1 - i / this.ghosts.length);
    }

    const showAurora = state.mode === "aurora";
    const workers = state.routing.flat.filter((w) => w.id !== "A");
    for (let i = 0; i < this.aurora.length; i++) {
      const mesh = this.aurora[i];
      const w = workers[i];
      if (!showAurora || !w) {
        mesh.visible = false;
        continue;
      }
      const ax = new THREE.Vector3(w.axis[0], w.axis[1], w.axis[2]).normalize();
      alignY(ax, tmpQ);
      mesh.visible = true;
      mesh.quaternion.copy(tmpQ);
      mesh.position.copy(ax).multiplyScalar(Math.cos(w.aperture));
      mesh.scale.setScalar(Math.sin(w.aperture));
      mesh.material.color.set(w.expanded ? PAPER : ICE);
      mesh.material.opacity = w.expanded ? 0.45 : 0.2;
    }

    if (optical) {
      const lp = this.lasers.geometry.attributes.position;
      lp.array.fill(0);
      let k = 0;
      for (const o of state.obs) {
        if (!o.laser) continue;
        const p = o.p;
        lp.array[k++] = 0; lp.array[k++] = 0; lp.array[k++] = 0;
        lp.array[k++] = p[0] * 1.15; lp.array[k++] = p[1] * 1.15; lp.array[k++] = p[2] * 1.15;
      }
      lp.needsUpdate = true;
    }

    const showStereo = morph > 0.12;
    this.stereoPlane.material.opacity = showStereo ? morph * 0.18 : 0;
    alignY(state.displayAxis, tmpQ);
    this.stereoGroup.quaternion.copy(tmpQ);
    this.stereoGroup.position.copy(state.displayAxis).multiplyScalar(lerp(0, 0.02, morph));
    for (let i = 0; i < this.stereoRings.length; i++) {
      const layer = layers[i];
      const ring = this.stereoRings[i];
      if (!layer || morph < 0.12) {
        ring.visible = false;
        continue;
      }
      ring.visible = true;
      const r = capRadiusOnPlane(layer.aperture) * 0.72;
      ring.scale.set(r, 1, r);
      ring.material.opacity = 0.15 + i * 0.1;
    }

    void dt;
  }
}

function wireUi() {
  document.querySelectorAll("#modes button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#modes button").forEach((b) => b.classList.remove("on"));
      btn.classList.add("on");
      state.mode = btn.dataset.mode;
      if (state.mode === "optical") setField("optical");
      else if (state.fieldKind === "optical") setField("sphere");
      if (state.mode === "stereo") {
        state.targetMorph = 1;
      } else if (state.mode === "field" || state.mode === "attention" || state.mode === "aurora" || state.mode === "memory") {
        if (state.targetMorph > 0.85) state.targetMorph = 0;
      }
      if (state.mode === "attention") recompute(true);
    });
  });

  $("morph").addEventListener("input", (e) => {
    state.targetMorph = Number(e.target.value) / 1000;
  });
  $("aperture").addEventListener("input", (e) => {
    state.aperture = Number(e.target.value) / 100;
    recompute(true);
  });
  $("depth").addEventListener("input", (e) => {
    state.depth = Number(e.target.value);
    recompute(true);
  });
  $("lambda").addEventListener("input", (e) => {
    state.lambda = Number(e.target.value) / 100;
    recompute(true);
  });
  $("temperature").addEventListener("input", (e) => {
    state.temperature = Number(e.target.value) / 100;
    recompute(true);
  });

  $("play").addEventListener("click", () => {
    state.playing = !state.playing;
    $("play").textContent = state.playing ? "PAUSE" : "PLAY";
    $("play").classList.toggle("on", state.playing);
  });
  $("aim").addEventListener("click", () => {
    state.aiming = !state.aiming;
    document.body.classList.toggle("aiming", state.aiming);
    $("aim").classList.toggle("on", state.aiming);
  });
  $("adaptive").addEventListener("click", () => {
    state.adaptive = !state.adaptive;
    $("adaptive").classList.toggle("on", state.adaptive);
    recompute(true);
  });
  $("collapse").addEventListener("click", () => {
    state.pulse = 1;
    const last = state.hierarchy.layers.at(-1);
    if (last) {
      state.aperture = clamp(last.aperture * 0.84, 0.12, 0.9);
      $("aperture").value = String(Math.round(state.aperture * 100));
      recompute(true);
    }
  });
  $("reset").addEventListener("click", () => {
    state.seed = (state.seed * 1103515245 + 12345) >>> 0;
    state.rng = mulberry32(state.seed);
    setField(state.fieldKind, state.seed);
    state.pulse = 1;
  });

  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input")) return;
    const map = { Digit1: "field", Digit2: "stereo", Digit3: "attention", Digit4: "aurora", Digit5: "memory", Digit6: "optical" };
    if (map[e.code]) document.querySelector(`[data-mode="${map[e.code]}"]`)?.click();
    if (e.code === "Space") {
      e.preventDefault();
      $("play").click();
    }
    if (e.code === "KeyA") $("aim").click();
    if (e.code === "KeyR") $("reset").click();
    if (e.code === "KeyC") $("collapse").click();
  });
}

setField("sphere");
wireUi();
const instrument = new Instrument($("stage"));
window.__cones = { state, instrument };
