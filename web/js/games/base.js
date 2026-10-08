// The game interface (port of game/base.py). The shell owns everything around
// a game - menu, countdown, pause, game over - and talks to it only through this.
//
//   static id, title, blurb      menu entry; games without a title stay off the menu
//   static menu = true           false keeps a game off the welcome screen
//   static tip                   countdown hint: what the player should have in frame
//   static async preload()       load art once at boot (the shell awaits it before the menu)
//   constructor(opts)            { width, height, mirrored, seed, art } - a FRESH instance
//                                for every start and replay
//   update(controls, dt)         controls from core/controls.js; clamp dt yourself
//   render(ctx, frame)           ctx: the 1280x720 2D context, already holding the live
//                                mirrored camera image; draw over it, cover it, or blend.
//                                frame: that camera image as a canvas (for picture-in-picture)
//   phase                        "game_over" when the game ends (any other string otherwise)
//   reset(), onResume()          onResume resets trackers / gesture history after a pause
//   swapHand()                   optional, for games with a dominant hand
//
// Keep the bottom-left footer (y > ~700) and the bottom-right PAUSE button
// (x 1090-1258, y 646-698) clear of important art. Score top-left or top-centre.

export class Game {
  static id = "game";
  static title = "";
  static blurb = "";
  static menu = true;
  static tip = "Stand back so your upper body is in frame";

  static async preload() {}

  constructor({ width = 1280, height = 720, mirrored = true, seed = null } = {}) {
    this.width = width;
    this.height = height;
    this.mirrored = mirrored;
    this.seed = seed;
    this.phase = "playing";
  }

  update(_controls, _dt) {}
  render(_ctx, _frame) {}
  reset() {}
  onResume() {}
}
