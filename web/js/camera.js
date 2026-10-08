// Webcam capture into a mirrored 1280x720 frame (the browser side of camera.py).
//
// Selfie view: the image is mirrored before anything sees it - the model, the
// games and the player - exactly like the desktop app. The model labels limbs
// as they appear, so the player's real right hand is left_wrist (hand.js).
//
// Camera choice follows the desktop rule: the built-in camera by default, never
// an iPhone (Continuity Camera) unless the player picks it.

import { loadImage } from "./core/gfx.js";

const PREF_KEY = "kp.camera";

export class Camera {
  constructor(width = 1280, height = 720) {
    this.width = width;
    this.height = height;
    this.video = document.createElement("video");
    this.video.playsInline = true;
    this.video.muted = true;
    this.frame = document.createElement("canvas");
    this.frame.width = width;
    this.frame.height = height;
    this.fctx = this.frame.getContext("2d", { alpha: false });
    this.stream = null;
    this.deviceId = null;
    this.label = "";
    this.frames = 0; // distinct camera frames seen
    this.lastFrameTime = 0;
    this._lastVideoTime = -1;
    this._rvfc = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
    this.blankSince = null;
  }

  static async list() {
    const all = await navigator.mediaDevices.enumerateDevices();
    return all.filter((d) => d.kind === "videoinput");
  }

  async open(deviceId = null) {
    this.close();
    const saved = deviceId ?? safeGet(PREF_KEY);
    // Ask for 60 fps: the camera's frame rate caps how often the model gets a
    // new image, and the GPU path (~20-25 ms) can keep up with ~40. Cameras
    // that only do 30 (most built-in webcams at 720p) just deliver 30.
    const video = { width: { ideal: this.width }, height: { ideal: this.height }, frameRate: { ideal: 60 } };
    let stream = await getStream(saved ? { ...video, deviceId: { exact: saved } } : { ...video, facingMode: "user" })
      .catch((err) => (saved ? getStream({ ...video, facingMode: "user" }) : Promise.reject(err)));

    // Labels are only visible after permission. If the browser picked a phone
    // and a built-in camera exists, switch to the built-in one.
    if (!deviceId && !saved) {
      const track = stream.getVideoTracks()[0];
      if (isPhone(track.label)) {
        const builtIn = (await Camera.list()).find((d) => !isPhone(d.label) && d.deviceId);
        if (builtIn) {
          stream.getTracks().forEach((t) => t.stop());
          stream = await getStream({ ...video, deviceId: { exact: builtIn.deviceId } });
        }
      }
    }

    this.stream = stream;
    const track = stream.getVideoTracks()[0];
    this.deviceId = track.getSettings().deviceId ?? null;
    this.label = track.label;
    this.fps = track.getSettings().frameRate ?? null; // what the camera actually agreed to
    this.video.srcObject = stream;
    await this.video.play();
    if (this._rvfc) this._watch();
    return this;
  }

  /** Remember an explicit choice from the camera picker. */
  async choose(deviceId) {
    safeSet(PREF_KEY, deviceId);
    return this.open(deviceId);
  }

  close() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  _watch() {
    const tick = () => {
      if (!this.stream) return;
      this.frames++;
      this.lastFrameTime = performance.now() / 1000;
      this.video.requestVideoFrameCallback(tick);
    };
    this.video.requestVideoFrameCallback(tick);
  }

  get ready() {
    return this.video.readyState >= 2 && this.video.videoWidth > 0;
  }

  /**
   * Draw the newest camera image, mirrored and cover-cropped to 16:9, into
   * `this.frame`. Returns true if it is a new camera frame since the last call.
   */
  grab() {
    if (!this.ready) return false;
    let fresh;
    if (this._rvfc) {
      fresh = this.frames !== this._grabbed;
      this._grabbed = this.frames;
    } else {
      fresh = this.video.currentTime !== this._lastVideoTime;
      this._lastVideoTime = this.video.currentTime;
      if (fresh) this.lastFrameTime = performance.now() / 1000;
    }
    if (!fresh) return false;
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    const k = Math.max(this.width / vw, this.height / vh);
    const sw = this.width / k, sh = this.height / k;
    const sx = (vw - sw) / 2, sy = (vh - sh) / 2;
    const g = this.fctx;
    g.save();
    g.translate(this.width, 0);
    g.scale(-1, 1);
    g.drawImage(this.video, sx, sy, sw, sh, 0, 0, this.width, this.height);
    g.restore();
    return true;
  }

  /** Seconds since the camera last delivered a frame. */
  age(now = performance.now() / 1000) {
    return this.lastFrameTime ? now - this.lastFrameTime : Infinity;
  }
}

function getStream(video) {
  return navigator.mediaDevices.getUserMedia({ video, audio: false });
}

export function isPhone(label = "") {
  return /iphone|ipad|continuity|desk view/i.test(label);
}

function safeGet(k) {
  try {
    return localStorage.getItem(k);
  } catch {
    return null;
  }
}

function safeSet(k, v) {
  try {
    localStorage.setItem(k, v);
  } catch {
    /* private mode */
  }
}

/**
 * A still image or video file standing in for the webcam (development and
 * demos: `?source=dev/people.jpg`). Same surface as Camera.
 */
export class FileSource {
  constructor(url, width = 1280, height = 720) {
    this.url = url;
    this.width = width;
    this.height = height;
    this.frame = document.createElement("canvas");
    this.frame.width = width;
    this.frame.height = height;
    this.fctx = this.frame.getContext("2d", { alpha: false });
    this.label = `file: ${url}`;
    this.deviceId = null;
    this.lastFrameTime = 0;
    this._last = 0;
  }

  async open() {
    if (/\.(png|jpe?g|webp|gif)(\?|$)/i.test(this.url)) {
      this.media = await loadImage(this.url);
      this.mw = this.media.naturalWidth;
      this.mh = this.media.naturalHeight;
    } else {
      this.media = document.createElement("video");
      Object.assign(this.media, { src: this.url, loop: true, muted: true, playsInline: true });
      await this.media.play();
      this.mw = this.media.videoWidth;
      this.mh = this.media.videoHeight;
    }
    return this;
  }

  get ready() {
    return !!this.media;
  }

  grab() {
    const now = performance.now() / 1000;
    if (!this.media || now - this._last < 1 / 30) return false;
    this._last = this.lastFrameTime = now;
    const k = Math.max(this.width / this.mw, this.height / this.mh);
    const sw = this.width / k, sh = this.height / k;
    const g = this.fctx;
    g.save();
    g.translate(this.width, 0);
    g.scale(-1, 1);
    g.drawImage(this.media, (this.mw - sw) / 2, (this.mh - sh) / 2, sw, sh, 0, 0, this.width, this.height);
    g.restore();
    return true;
  }

  age(now = performance.now() / 1000) {
    return this.lastFrameTime ? now - this.lastFrameTime : Infinity;
  }

  close() {}
}
