// Game pages: the start screen's showcase video.
// - Reduced motion: show the poster, never play.
// - Once the game starts (main.js hides #intro), stop decoding the video so
//   the GPU is free for the pose model; resume if the start screen returns.
(function () {
  var video = document.querySelector(".feed video");
  var intro = document.getElementById("intro");
  if (!video || !intro) return;
  var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)");

  function sync() {
    var show = !intro.hidden && !(still && still.matches);
    if (show) {
      if (video.paused) video.play().catch(function () {});
    } else if (!video.paused || video.autoplay) {
      video.autoplay = false;
      video.pause();
      if (still && still.matches) video.load(); // back to the poster
    }
  }
  sync();
  new MutationObserver(sync).observe(intro, { attributes: true, attributeFilter: ["hidden"] });
  if (still && still.addEventListener) still.addEventListener("change", sync);
})();
