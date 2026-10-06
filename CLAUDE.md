# KP

Body-controlled games on a shared RF-DETR pose backbone. The full agent guide
is AGENTS.md (imported below). Before changing a subsystem, read its section in
AGENTS.md and the matching rationale in README.md.

@AGENTS.md

## Always
- New games start from `src/kpapp/game/template/` and are registered in `src/kpapp/game/__init__.py`.
- Run `uv run --with pytest pytest tests/ -q` before reporting work as done.
- Verify visually: render key moments to an image and look at it. Tests miss most visual bugs.
- Live camera runs go through the user's terminal; a sandboxed shell may be denied the camera.
- Original, procedural, deterministic art only. Unseeded randomness in games, with a `seed` arg for tests.
