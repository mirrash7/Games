// Seedable PRNG with the slice of Python's random.Random the games use.
// seed = null/undefined -> fresh entropy every game (games must not repeat).

export class Rng {
  constructor(seed = null) {
    let s = seed == null ? (Math.random() * 2 ** 32) >>> 0 : hashSeed(seed);
    this._s = s >>> 0;
    this._spare = null;
  }

  /** Uniform in [0, 1). mulberry32. */
  random() {
    let t = (this._s = (this._s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(a, b) { return a + (b - a) * this.random(); }

  /** Integer in [a, b], inclusive like Python's randint. */
  randint(a, b) { return a + Math.floor(this.random() * (b - a + 1)); }

  choice(arr) { return arr[Math.floor(this.random() * arr.length)]; }

  gauss(mu = 0, sigma = 1) {
    if (this._spare != null) {
      const v = this._spare;
      this._spare = null;
      return mu + sigma * v;
    }
    let u = 0;
    while (u <= 1e-12) u = this.random();
    const r = Math.sqrt(-2 * Math.log(u));
    const th = 2 * Math.PI * this.random();
    this._spare = r * Math.sin(th);
    return mu + sigma * r * Math.cos(th);
  }
}

function hashSeed(seed) {
  if (typeof seed === "number") return (Math.floor(seed) ^ 0x9e3779b9) >>> 0;
  let h = 2166136261;
  for (const ch of String(seed)) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return h >>> 0;
}
