// Decode raw RF-DETR keypoint-model outputs into poses.
//
// A line-for-line port of rfdetr's PostProcess for the keypoint preview
// checkpoint (models/postprocess.py), so the browser sees exactly the poses
// the desktop app does. web/tests/decode.test.mjs checks it against rf-detr's
// own decoder on a fixed image.
//
// Outputs of the exported ONNX graph:
//   dets      [1, Q, 4]        boxes, normalised cx, cy, w, h
//   labels    [1, Q, C]        class logits; C=2, class 0 is background
//   keypoints [1, Q, C*17, D]  per class 17 keypoints; D>=7:
//                              x, y (normalised), visibility logit, -, and the
//                              Cholesky factor of the keypoint precision
//                              (log l11, l21, log l22) used for score fusion

export const NUM_KEYPOINTS = 17;
export const KEYPOINTS_PER_CLASS = [0, 17]; // class 1 = person
export const TRACE_ALPHA = 0.2;
export const NUM_SELECT = 100;

const sigmoid = (v) => 1 / (1 + Math.exp(-v));

function logsumexp(values) {
  let m = -Infinity;
  for (const v of values) if (v > m) m = v;
  if (m === -Infinity) return -Infinity;
  let s = 0;
  for (const v of values) s += Math.exp(v - m);
  return m + Math.log(s);
}

// Uncertainty of a detection's keypoints, as the log of the visibility-weighted
// mean trace of each keypoint's covariance (rfdetr _keypoint_log_mean_trace).
function logMeanTrace(kp, base, n, D) {
  const weighted = [];
  const logW = [];
  for (let k = 0; k < n; k++) {
    const o = base + k * D;
    const logL11 = kp[o + 4];
    const l21 = kp[o + 5];
    const logL22 = kp[o + 6];
    const wFind = sigmoid(kp[o + 2]);
    const logT1 = -2 * logL11;
    const logT2 = -2 * logL22;
    const logT3 = 2 * Math.log(Math.max(Math.abs(l21), 1e-12)) + logT1 + logT2;
    const lw = Math.log(Math.max(wFind, 1e-12));
    weighted.push(logsumexp([logT1, logT2, logT3]) + lw);
    logW.push(lw);
  }
  return logsumexp(weighted) - logsumexp(logW);
}

/**
 * @param {{dets: Float32Array, labels: Float32Array, keypoints: Float32Array,
 *          Q: number, C: number, slots: number, D: number}} raw
 * @param {number} width  source image width the keypoints are scaled to
 * @param {number} height source image height
 * @param {number} threshold minimum fused score (the desktop app uses 0.5)
 * @returns {{score:number, box:number[], xy:Float32Array, conf:Float32Array}[]}
 *          sorted by score, highest first. xy is [x0,y0,x1,y1,...] (17 pairs).
 */
export function decodePoses(raw, width, height, threshold = 0.5) {
  const { dets, labels, keypoints, Q, C, slots, D } = raw;
  const maxK = Math.max(...KEYPOINTS_PER_CLASS);
  if (slots !== KEYPOINTS_PER_CLASS.length * maxK) {
    throw new Error(`unexpected keypoint slots ${slots}`);
  }

  // Top-k over every (query, class) probability; ties broken by flat index,
  // matching rfdetr's stable descending argsort.
  const cand = [];
  for (let q = 0; q < Q; q++) {
    for (let c = 0; c < C; c++) cand.push({ p: sigmoid(labels[q * C + c]), idx: q * C + c });
  }
  cand.sort((a, b) => b.p - a.p || a.idx - b.idx);
  const selected = cand.slice(0, Math.min(NUM_SELECT, cand.length));

  const poses = [];
  for (const { p, idx } of selected) {
    const q = Math.floor(idx / C);
    const c = idx % C;
    const n = KEYPOINTS_PER_CLASS[c] ?? 0;
    // Every class goes through score fusion; zero-keypoint classes fuse with a
    // trace of 0, which maps any score to s/(1+s) < 0.5 - why background
    // detections never pass the usual threshold.
    const base = (q * slots + c * maxK) * D;
    const lmt = n > 0 && D >= 7 ? logMeanTrace(keypoints, base, n, D) : 0;
    const score = Math.min(sigmoid(Math.log(p) - TRACE_ALPHA * lmt), 1 - 1e-7);
    if (n <= 0 || score <= threshold) continue;

    const cx = dets[q * 4], cy = dets[q * 4 + 1], w = dets[q * 4 + 2], h = dets[q * 4 + 3];
    const clampX = (v) => Math.min(Math.max(v, 0), width);
    const clampY = (v) => Math.min(Math.max(v, 0), height);
    const box = [clampX((cx - w / 2) * width), clampY((cy - h / 2) * height),
                 clampX((cx + w / 2) * width), clampY((cy + h / 2) * height)];
    const xy = new Float32Array(NUM_KEYPOINTS * 2);
    const conf = new Float32Array(NUM_KEYPOINTS);
    for (let k = 0; k < n; k++) {
      const o = base + k * D;
      xy[2 * k] = keypoints[o] * width;
      xy[2 * k + 1] = keypoints[o + 1] * height;
      conf[k] = sigmoid(keypoints[o + 2]);
    }
    poses.push({ score, box, xy, conf });
  }
  poses.sort((a, b) => b.score - a.score);
  return poses;
}
