// The landing hero: a 3D keypoint skeleton dancing the real Raccoon Dance
// choreography (Neon Nights, no audio) inside a tracked bounding box.
//
// The dance is the game's own: moves.json + song.json parsed by
// js/games/dance/chart.js, posed by Song.coachPose(beat), and turned into
// joints by pose.js forward() (shoulder widths, y down). Here those joints
// become the 17 COCO keypoints a pose model reports, with a little cosmetic
// depth, drawn as glowing spheres and bones.
//
// three.js is imported lazily (from the page's importmap), so Node can import
// the pure helpers below for tests. Performance: devicePixelRatio is capped,
// rendering stops when the hero is off-screen or the tab is hidden, and
// prefers-reduced-motion renders a single still frame.

import { forward, lerpPose, sideKeypoints, smoothstep } from "../js/games/dance/pose.js";
import { idleGroove, parseLibrary, parseSong } from "../js/games/dance/chart.js";

export const SONG = "neon_nights";
// Verse 1 + chorus 1 + verse 2 (beats), looped with a crossfade.
export const LOOP = Object.freeze({ start: 14, end: 110, fade: 1.5 });
export const STILL_BEAT = 48.0; // reduced motion: Disco Point, on its beat
export const MAX_DPR = 1.75;

/** Seconds since start -> beat inside the loop. */
export function loopBeat(t, bpm, loop = LOOP) {
  const len = loop.end - loop.start;
  const b = (t * bpm) / 60;
  return loop.start + (((b % len) + len) % len);
}

/** The coach's pose at a loop beat, crossfading the loop's end into its start. */
export function heroPose(song, beat, loop = LOOP) {
  const at = (b) => (song ? song.coachPose(b) : idleGroove(b));
  const into = beat - (loop.end - loop.fade);
  if (into <= 0) return at(beat);
  return lerpPose(at(beat), at(loop.start - loop.fade + into), smoothstep(into / loop.fade));
}

/** The placed move under a beat: {label, beat, end} or null. */
export function moveAt(song, beat) {
  if (!song) return null;
  for (const p of song.chart) if (p.beat <= beat && beat < p.end) return { label: p.move.label, beat: p.beat, end: p.end };
  return null;
}

export const COCO = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
];
const I = Object.fromEntries(COCO.map((n, i) => [n, i]));

/** Bones between COCO indices (the face is dots only, as the model reports it). */
export const BONES = [
  ["left_shoulder", "right_shoulder"], ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"], ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"], ["left_hip", "right_hip"], ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"], ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
].map(([a, b]) => [I[a], I[b]]);

/**
 * A dance pose -> 17 COCO keypoints [x, y, z] in shoulder widths, y up,
 * origin at the rest hip centre. Mirrored like the game's camera feed, so
 * the screen-right limbs are left_* (AGENTS.md section 6). z is cosmetic depth.
 */
export function keypoints17(pose) {
  const j = forward(pose);
  const sides = sideKeypoints(true);
  const out = new Array(17);
  const put = (name, p, z) => { out[I[name]] = [p[0], -p[1], z]; };
  // Face around the head centre, turned with the head.
  const [hx, hy] = j.head;
  const tilt = Math.atan2(j.head[0] - j.neck[0], j.neck[1] - j.head[1]);
  const c = Math.cos(tilt), s = Math.sin(tilt);
  const face = (dx, dy) => [hx + dx * c - dy * s, hy + dx * s + dy * c];
  put("nose", face(0, 0.06), 0.28);
  for (const [side, sx] of [["l", -1], ["r", 1]]) {
    const k = sides[side];
    put(`${k}_eye`, face(sx * 0.1, -0.05), 0.22);
    put(`${k}_ear`, face(sx * 0.22, 0.0), 0.0);
    put(`${k}_shoulder`, j[`${side}_shoulder`], 0.0);
    put(`${k}_elbow`, j[`${side}_elbow`], 0.12);
    put(`${k}_wrist`, j[`${side}_wrist`], 0.26);
    put(`${k}_hip`, j[`${side}_hip`], 0.0);
    put(`${k}_knee`, j[`${side}_knee`], 0.14 + 0.5 * Math.max(0, j.hip[1]));
    put(`${k}_ankle`, j[`${side}_ankle`], 0.0);
  }
  return out;
}

/** Axis-aligned bounds of points: {min: [x,y,z], max: [x,y,z]}. */
export function bounds(points) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let k = 0; k < 3; k++) {
      if (p[k] < min[k]) min[k] = p[k];
      if (p[k] > max[k]) max[k] = p[k];
    }
  }
  return { min, max };
}

/** A believable detection score: 0.95-0.99, drifting slowly. */
export function confidenceAt(t) {
  const v = 0.972 + 0.012 * Math.sin(t * 0.9) + 0.006 * Math.sin(t * 2.3 + 1.0);
  return Math.min(0.99, Math.max(0.95, v)).toFixed(2);
}

// --- browser ----------------------------------------------------------------

const COLORS = {
  bg: 0x0f0a1c,
  joint: 0xf1e6ff,
  bone: 0xa874f2,
  brand: 0x8033eb,
  lav: 0xc29af6,
  grid: 0x6f4bb0,
  orange: 0xd96210,
};

async function loadSong(base) {
  const get = async (p) => {
    const r = await fetch(base + p);
    if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
    return r.json();
  };
  const [moves, song] = await Promise.all([get("moves.json"), get(`songs/${SONG}/song.json`)]);
  return parseSong(song, SONG, parseLibrary(moves, "assets/dance/moves.json"));
}

function webglAvailable() {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

/** Radial glow texture (white, alpha falls off), drawn once. */
function glowTexture(THREE, size = 128) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.35)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** Vertical band texture for the scan sweep: bright core line, soft tail above. */
function scanTexture(THREE) {
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 128;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 128);
  grad.addColorStop(0, "rgba(255,255,255,0)");
  grad.addColorStop(0.86, "rgba(255,255,255,0.16)");
  grad.addColorStop(0.97, "rgba(255,255,255,0.9)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 128);
  return new THREE.CanvasTexture(c);
}

/** The fallback when there's no WebGL (or three.js didn't load): the hero video, else the CSS gradient. */
export function heroFallback(hero, { reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches } = {}) {
  hero.classList.add("hero-static");
  const slot = hero.querySelector("[data-hero-video]");
  if (!slot || slot.childElementCount) return;
  const v = document.createElement("video");
  Object.assign(v, { muted: true, loop: true, playsInline: true, autoplay: true, preload: "metadata" });
  v.setAttribute("aria-hidden", "true");
  v.poster = "media/hero.jpg";
  if (reducedMotion) v.autoplay = false; // the poster stays as a still
  for (const [src, type] of [["media/hero.webm", "video/webm"], ["media/hero.mp4", "video/mp4"]]) {
    const s = document.createElement("source");
    s.src = src;
    s.type = type;
    v.append(s);
  }
  // The last source failing means no video at all: keep the gradient.
  v.lastElementChild.addEventListener("error", () => v.remove());
  slot.append(v);
}

/**
 * Start the hero on a canvas inside #hero. Resolves once the first frame is
 * drawn. Options: reducedMotion, assetBase.
 */
export async function startHero(canvas, { reducedMotion = false, assetBase = "assets/dance/" } = {}) {
  const hero = canvas.closest("#hero") ?? canvas.parentElement;
  if (!webglAvailable()) {
    heroFallback(hero, { reducedMotion });
    return null;
  }
  let THREE, EffectComposer, RenderPass, UnrealBloomPass, OutputPass;
  try {
    [THREE, { EffectComposer }, { RenderPass }, { UnrealBloomPass }, { OutputPass }] = await Promise.all([
      import("three"),
      import("three/addons/postprocessing/EffectComposer.js"),
      import("three/addons/postprocessing/RenderPass.js"),
      import("three/addons/postprocessing/UnrealBloomPass.js"),
      import("three/addons/postprocessing/OutputPass.js"),
    ]);
  } catch (err) {
    console.warn("hero: three.js didn't load", err);
    heroFallback(hero, { reducedMotion });
    return null;
  }
  const song = await loadSong(assetBase).catch((err) => {
    console.warn("hero: dance chart didn't load, idling", err);
    return null;
  });
  const bpm = song?.bpm ?? 120;

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
  } catch (err) {
    console.warn("hero: WebGL context failed", err);
    heroFallback(hero, { reducedMotion });
    return null;
  }
  renderer.setClearColor(COLORS.bg, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(COLORS.bg, 9, 30);
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 80);

  // Everything that dances lives in `rig`, which stands on the floor.
  const FLOOR = -1.78;
  const rig = new THREE.Group();
  scene.add(rig);

  // Floor: a perspective grid fading into the fog, plus a glow pad.
  const grid = new THREE.Mesh(
    new THREE.PlaneGeometry(60, 60),
    new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uColor: { value: new THREE.Color(COLORS.lav) },
        uOpacity: { value: 0.5 },
        uRadius: { value: 15.0 },
      },
      vertexShader: `
        varying vec2 vXZ;
        void main() {
          vec4 w = modelMatrix * vec4(position, 1.0);
          vXZ = w.xz;
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: `
        uniform vec3 uColor;
        uniform float uOpacity;
        uniform float uRadius;
        varying vec2 vXZ;
        float lines(vec2 p) {
          vec2 g = abs(fract(p - 0.5) - 0.5) / fwidth(p);
          return 1.0 - min(min(g.x, g.y), 1.0);
        }
        void main() {
          float minor = lines(vXZ * 2.0) * 0.35;
          float major = lines(vXZ * 0.5);
          float fade = 1.0 - smoothstep(0.15, 1.0, length(vXZ) / uRadius);
          float a = max(minor, major) * fade * uOpacity;
          if (a < 0.003) discard;
          gl_FragColor = vec4(uColor, a);
        }`,
    }),
  );
  grid.rotation.x = -Math.PI / 2;
  grid.position.y = FLOOR;
  scene.add(grid);
  const glow = glowTexture(THREE);
  const pad = new THREE.Mesh(
    new THREE.PlaneGeometry(4.2, 4.2),
    new THREE.MeshBasicMaterial({ map: glow, color: COLORS.brand, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }),
  );
  pad.rotation.x = -Math.PI / 2;
  pad.position.y = FLOOR + 0.01;
  rig.add(pad);
  // A soft violet halo behind the dancer.
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color: 0x3a1d6e, transparent: true, opacity: 0.9, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
  halo.scale.set(9, 9, 1);
  halo.position.set(0, 0.2, -2.5);
  rig.add(halo);

  // Skeleton: instanced joint spheres and bone cylinders.
  const jointMat = new THREE.MeshBasicMaterial({ color: COLORS.joint });
  const joints = new THREE.InstancedMesh(new THREE.SphereGeometry(0.052, 16, 12), jointMat, 17);
  const boneGeo = new THREE.CylinderGeometry(0.018, 0.018, 1, 8, 1, true);
  boneGeo.translate(0, 0.5, 0); // pivot at one end
  const bones = new THREE.InstancedMesh(boneGeo, new THREE.MeshBasicMaterial({ color: COLORS.bone }), BONES.length);
  // Bigger, dimmer halos on every joint read as "keypoint" at a glance.
  const haloMat = new THREE.MeshBasicMaterial({ color: COLORS.brand, transparent: true, opacity: 0.16, depthWrite: false, blending: THREE.AdditiveBlending });
  const jointHalos = new THREE.InstancedMesh(new THREE.SphereGeometry(0.12, 12, 8), haloMat, 17);
  // The head outline (the coach's head), always facing the camera.
  const headRing = new THREE.Mesh(
    new THREE.RingGeometry(0.29, 0.31, 48),
    new THREE.MeshBasicMaterial({ color: COLORS.lav, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }),
  );
  for (const m of [joints, bones, jointHalos]) m.frustumCulled = false;
  rig.add(bones, joints, jointHalos, headRing);

  // Wrist trails: where the hands have been.
  const TRAIL = 36;
  const trails = [I.left_wrist, I.right_wrist].map((idx) => {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(TRAIL * 3);
    const col = new Float32Array(TRAIL * 3);
    const c = new THREE.Color(COLORS.lav);
    for (let i = 0; i < TRAIL; i++) {
      const a = Math.pow(1 - i / (TRAIL - 1), 1.6);
      col.set([c.r * a, c.g * a, c.b * a], i * 3);
    }
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    line.frustumCulled = false;
    rig.add(line);
    return { idx, pos, geo, filled: false };
  });

  // Bounding box: a 3D wireframe box plus bright corner ticks on its front face.
  const boxGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
  boxGeo.translate(0.5, 0.5, 0.5);
  const box = new THREE.LineSegments(boxGeo, new THREE.LineBasicMaterial({ color: 0x9a5cf5, transparent: true, opacity: 0.9 }));
  rig.add(box);
  const tickPts = [];
  for (const [x, y, dx, dy] of [[0, 0, 1, 1], [1, 0, -1, 1], [0, 1, 1, -1], [1, 1, -1, -1]]) {
    tickPts.push(x, y, 1, x + dx * 0.12, y, 1, x, y, 1, x, y + dy * 0.08, 1);
  }
  const tickGeo = new THREE.BufferGeometry();
  tickGeo.setAttribute("position", new THREE.Float32BufferAttribute(tickPts, 3));
  const ticks = new THREE.LineSegments(tickGeo, new THREE.LineBasicMaterial({ color: 0xffffff }));
  box.add(ticks);
  // Scan sweep through the box.
  const scan = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({ map: scanTexture(THREE), color: COLORS.lav, transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending }),
  );
  rig.add(scan);

  // Dust: slow particles drifting up through the scene.
  const DUST = 260;
  const dustPos = new Float32Array(DUST * 3);
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < DUST; i++) dustPos.set([(rnd() - 0.5) * 16, FLOOR + rnd() * 7, (rnd() - 0.5) * 10 - 1], i * 3);
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
  const dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: COLORS.lav, size: 0.035, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
  scene.add(dust);

  // Post: bloom on a multisampled target.
  const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
  const composer = new EffectComposer(renderer, target);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.6, 0.45, 0.32);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  // HTML overlays: the detection label and the HUD.
  const label = hero.querySelector("[data-hero-label]");
  const labelScore = label?.querySelector("[data-score]");
  const hudMove = hero.querySelector("[data-hud-move]");
  const hudBeat = hero.querySelector("[data-hud-beat]");
  const hudSong = hero.querySelector("[data-hud-song]");
  if (hudSong && song) hudSong.textContent = `${song.title} · ${Math.round(bpm)} BPM`;

  // Layout: where the dancer sits in the frame, and how far the camera is.
  // Wide screens: right of the headline. Tall screens: in the space below it.
  const FIG_H = 4.4, FIG_W = 4.8; // shoulder widths: floor to label, arms out
  const copyEl = hero.querySelector(".hero-copy");
  let width = 1, height = 1, dist = 12;
  const resize = () => {
    const r = canvas.getBoundingClientRect();
    width = Math.max(1, Math.round(r.width));
    height = Math.max(1, Math.round(r.height));
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    renderer.setPixelRatio(dpr);
    renderer.setSize(width, height, false);
    composer.setPixelRatio(dpr);
    composer.setSize(width, height);
    const aspect = width / height;
    camera.aspect = aspect;
    const fit = 2 * Math.tan((camera.fov * Math.PI) / 360); // visible height per unit of distance
    let cx, cy, regionW, regionH;
    const copy = copyEl?.getBoundingClientRect();
    if (width < 900 || aspect < 1.05) {
      const top = copy ? copy.bottom - r.top + 40 : height * 0.5;
      const bottom = height - 84; // room for the HUD
      regionH = Math.max(220, bottom - top);
      regionW = width - 32;
      cx = width / 2;
      cy = top + regionH / 2;
    } else {
      const left = copy ? copy.right - r.left + 24 : width * 0.45;
      regionW = Math.max(320, width - left - 24);
      regionH = (height - 72) * 0.74;
      cx = Math.min(left + regionW / 2, width * 0.74);
      cy = 72 + (height - 72) / 2 + 8;
    }
    // Narrow regions (phones, small laptops): let outstretched arms brush the edges.
    const figW = regionW < 560 ? 3.8 : FIG_W;
    dist = Math.max(FIG_H / (fit * (regionH / height)), figW / (fit * aspect * (regionW / width)));
    scene.fog.near = dist * 0.85;
    scene.fog.far = dist * 2.6;
    camera.setViewOffset(width, height, width / 2 - cx, height / 2 - cy, width, height);
    camera.updateProjectionMatrix();
  };
  resize();
  new ResizeObserver(resize).observe(canvas);

  // Pointer parallax (desktop): small orbit offsets, eased.
  const pointer = { x: 0, y: 0, ex: 0, ey: 0 };
  if (!reducedMotion) {
    addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse") return;
      pointer.x = (e.clientX / innerWidth) * 2 - 1;
      pointer.y = (e.clientY / innerHeight) * 2 - 1;
    }, { passive: true });
  }

  const m4 = new THREE.Matrix4();
  const v = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  const q = new THREE.Quaternion();
  const sc = new THREE.Vector3();
  const tmp = new THREE.Vector3();
  const look = new THREE.Vector3(0, 0.15, 0);
  let clock = 0, last = performance.now(), frameMs = 0, hudAt = 0, lastMove = "";

  const draw = (t, beat) => {
    // Pose -> keypoints.
    const pts = keypoints17(heroPose(song, beat));
    for (let i = 0; i < 17; i++) {
      const p = pts[i];
      const s = i < 5 ? 0.7 : 1.0; // face dots smaller
      m4.compose(v.set(p[0], p[1], p[2]), q.identity(), sc.set(s, s, s));
      joints.setMatrixAt(i, m4);
      jointHalos.setMatrixAt(i, m4);
    }
    BONES.forEach(([a, b], k) => {
      const pa = pts[a], pb = pts[b];
      const dir = tmp.set(pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]);
      const len = dir.length();
      q.setFromUnitVectors(up, dir.normalize());
      m4.compose(v.set(pa[0], pa[1], pa[2]), q, sc.set(1, len, 1));
      bones.setMatrixAt(k, m4);
    });
    joints.instanceMatrix.needsUpdate = bones.instanceMatrix.needsUpdate = jointHalos.instanceMatrix.needsUpdate = true;
    const le = pts[I.left_ear], re = pts[I.right_ear];
    headRing.position.set((le[0] + re[0]) / 2, (le[1] + re[1]) / 2 + 0.01, 0.05);
    headRing.quaternion.copy(camera.quaternion);

    // Trails.
    for (const tr of trails) {
      const p = pts[tr.idx];
      if (!tr.filled) {
        for (let i = 0; i < TRAIL; i++) tr.pos.set(p, i * 3);
        tr.filled = true;
      } else {
        tr.pos.copyWithin(3, 0, (TRAIL - 1) * 3);
        tr.pos.set(p, 0);
      }
      tr.geo.attributes.position.needsUpdate = true;
    }

    // Bounding box around the keypoints, padded like a detector's box.
    const b = bounds(pts);
    const pad = 0.16;
    const bx = b.min[0] - pad, by = Math.min(b.min[1] - 0.06, FLOOR + 0.02), bz = -0.35;
    const bw = b.max[0] - b.min[0] + 2 * pad, bh = b.max[1] + 0.34 - by, bd = 0.85;
    box.position.set(bx, by, bz);
    box.scale.set(bw, bh, bd);
    // Scan: sweeps top to bottom every 2.8 s.
    const sweep = reducedMotion ? 0.35 : (t / 2.8) % 1;
    scan.scale.set(bw, Math.min(0.55, bh), 1);
    scan.position.set(bx + bw / 2, by + bh - sweep * bh - scan.scale.y / 2, bz + bd / 2);
    scan.material.opacity = 0.55 * Math.sin(Math.PI * Math.min(1, sweep * 1.15));

    // Camera: slow drift around the dancer plus pointer parallax.
    pointer.ex += (pointer.x - pointer.ex) * 0.05;
    pointer.ey += (pointer.y - pointer.ey) * 0.05;
    const yaw = (reducedMotion ? 0.32 : 0.38 * Math.sin(t * 0.13) + 0.12) + pointer.ex * 0.18;
    const pitch = 0.17 + (reducedMotion ? 0 : 0.04 * Math.sin(t * 0.09)) + pointer.ey * 0.06;
    camera.position.set(Math.sin(yaw) * dist, look.y + Math.sin(pitch) * dist + 0.3, Math.cos(yaw) * dist);
    camera.lookAt(look);
    dust.rotation.y = t * 0.01;
    if (!reducedMotion) {
      const arr = dustGeo.attributes.position.array;
      for (let i = 1; i < arr.length; i += 3) {
        arr[i] += 0.0025;
        if (arr[i] > FLOOR + 7) arr[i] = FLOOR;
      }
      dustGeo.attributes.position.needsUpdate = true;
    }

    composer.render();

    // The detection label sits on the box's top-left front corner.
    if (label) {
      v.set(bx, by + bh, bz + bd).project(camera);
      const x = (v.x * 0.5 + 0.5) * width, y = (-v.y * 0.5 + 0.5) * height;
      label.style.transform = `translate3d(${x.toFixed(1)}px, ${(y - label.offsetHeight).toFixed(1)}px, 0)`;
      if (labelScore && t - hudAt > 0.25) labelScore.textContent = confidenceAt(t);
    }
    if (t - hudAt > 0.25 || hudAt === 0) {
      hudAt = t;
      const mv = moveAt(song, beat);
      const name = mv ? mv.label : "Groove";
      if (hudMove && name !== lastMove) { hudMove.textContent = name; lastMove = name; }
      if (hudBeat) hudBeat.textContent = String(Math.floor(beat)).padStart(3, "0");
      label?.classList.add("on");
    }
  };

  // Visibility: run only while the hero is on screen and the tab is visible.
  let onScreen = true, raf = 0;
  const running = () => onScreen && !document.hidden && !reducedMotion;
  const tick = (now) => {
    raf = 0;
    if (!running()) return;
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    clock += dt;
    const t0 = performance.now();
    draw(clock, loopBeat(clock, bpm));
    frameMs = frameMs * 0.95 + (performance.now() - t0) * 0.05;
    raf = requestAnimationFrame(tick);
  };
  const wake = () => {
    if (raf || !running()) return;
    last = performance.now();
    raf = requestAnimationFrame(tick);
  };
  new IntersectionObserver((entries) => {
    onScreen = entries.some((e) => e.isIntersecting);
    wake();
  }).observe(canvas);
  document.addEventListener("visibilitychange", wake);

  // Reduced motion: one still frame, redrawn on resize. Starting at the loop
  // start keeps the first frames of the animated version deterministic too.
  if (reducedMotion) {
    const still = () => draw(0, STILL_BEAT);
    still();
    new ResizeObserver(() => requestAnimationFrame(still)).observe(canvas);
  } else {
    clock = (2 * 60) / bpm; // open on the first move (beat 16), not the idle
    draw(clock, loopBeat(clock, bpm));
    wake();
  }
  hero.classList.add("hero-live");
  performance.mark?.("hero-live");
  const api = {
    get frameMs() { return frameMs; }, // CPU time per frame (smoothed)
    renderer,
    /** Average ms per frame including the GPU (readPixels waits for it). Debugging only. */
    bench(n = 60) {
      const gl = renderer.getContext(), px = new Uint8Array(4);
      const t0 = performance.now();
      for (let i = 0; i < n; i++) {
        draw(clock + i / 60, loopBeat(clock + i / 60, bpm));
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      }
      return (performance.now() - t0) / n;
    },
  };
  window.__hero = api; // for debugging and the perf check
  return api;
}
