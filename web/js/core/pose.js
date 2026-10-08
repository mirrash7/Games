// One detected person and the COCO-17 keypoint layout (port of config.py / inference.Pose).

export const KEYPOINT_NAMES = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
];

export const KP = Object.fromEntries(KEYPOINT_NAMES.map((n, i) => [n, i]));

// [a, b, css colour] edges for the debug skeleton.
export const SKELETON = [
  [KP.left_shoulder, KP.right_shoulder, "rgb(0,200,255)"],
  [KP.left_shoulder, KP.left_elbow, "rgb(255,220,0)"],
  [KP.left_elbow, KP.left_wrist, "rgb(255,220,0)"],
  [KP.right_shoulder, KP.right_elbow, "rgb(255,140,0)"],
  [KP.right_elbow, KP.right_wrist, "rgb(255,140,0)"],
  [KP.left_shoulder, KP.left_hip, "rgb(0,200,255)"],
  [KP.right_shoulder, KP.right_hip, "rgb(0,200,255)"],
  [KP.left_hip, KP.right_hip, "rgb(0,200,255)"],
  [KP.left_hip, KP.left_knee, "rgb(120,255,120)"],
  [KP.left_knee, KP.left_ankle, "rgb(120,255,120)"],
  [KP.right_hip, KP.right_knee, "rgb(60,200,60)"],
  [KP.right_knee, KP.right_ankle, "rgb(60,200,60)"],
  [KP.nose, KP.left_eye, "rgb(255,160,200)"],
  [KP.nose, KP.right_eye, "rgb(255,160,200)"],
  [KP.left_eye, KP.left_ear, "rgb(255,160,200)"],
  [KP.right_eye, KP.right_ear, "rgb(255,160,200)"],
];

export class Pose {
  /**
   * @param {Float32Array|number[]} xy 34 values: x0, y0, x1, y1, ... in pixels of the 1280x720 mirrored frame
   * @param {Float32Array|number[]} confidence 17 per-keypoint confidences
   */
  constructor(xy, confidence, score = 1, box = null) {
    this.xy = xy instanceof Float32Array ? xy : Float32Array.from(xy);
    this.confidence = confidence instanceof Float32Array ? confidence : Float32Array.from(confidence);
    this.score = score;
    this.box = box;
  }

  /** Keypoint `i` as [x, y] if it is confidently visible, else null. */
  point(i, minConf = 0.5) {
    if (!(this.confidence[i] >= minConf)) return null;
    return [this.xy[2 * i], this.xy[2 * i + 1]];
  }

  x(i) { return this.xy[2 * i]; }
  y(i) { return this.xy[2 * i + 1]; }

  static fromObject(o) {
    return new Pose(o.xy, o.conf ?? o.confidence, o.score ?? 1, o.box ?? null);
  }
}
