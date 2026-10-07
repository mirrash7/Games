"""Fruit Ninja: blade tracking, slice detection and the classic rules."""

from __future__ import annotations

import numpy as np
import pytest

from kpapp.config import KEYPOINT_NAMES, KP
from kpapp.controls import ControlState
from kpapp.game.fruitninja.art import Art, FruitArt
from kpapp.game.fruitninja.blade import Blade, blade_keypoint
from kpapp.game.fruitninja.entities import Fruit, Spawner
from kpapp.game.fruitninja.game import FruitNinjaGame, Phase, Rules
from kpapp.inference import Pose

W, H = 1280, 720
DT = 1 / 30


def stub_art() -> Art:
    spr = np.zeros((110, 110, 4), np.uint8)
    spr[:, :, 3] = 255
    fruits = [FruitArt(f"f{i}", spr, spr, spr, (50, 50, 220), 52, 1) for i in range(4)]
    return Art(fruits=fruits, bomb=spr, background=np.zeros((H, W, 3), np.uint8),
               splat=spr, flash=spr, life_full=spr, life_lost=spr)


def hand_pose(x: float, y: float, conf: float = 0.9) -> Pose:
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    xy[KP["left_wrist"]] = (x, y)
    c[KP["left_wrist"]] = conf
    for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        xy[KP[n]] = (640, 360)
        c[KP[n]] = 0.9
    return Pose(xy=xy, confidence=c, score=0.9)


def new_game(**kw) -> FruitNinjaGame:
    return FruitNinjaGame((W, H), art=stub_art(), **kw)


def swipe(game: FruitNinjaGame, x0: float, x1: float, y: float, step: float = 120.0) -> None:
    """A fast swing, sampled the way tracking delivers it.

    Recorded play never moved more than ~245px between samples; a single
    800px jump is what a tracking glitch looks like and is ignored by design.
    """
    n = max(1, int(np.ceil(abs(x1 - x0) / step)))
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        game.update(ControlState(present=True, pose=hand_pose(x, y)), DT)


# --- which hand is the blade ---


def test_mirrored_right_hand_is_the_left_wrist_keypoint():
    """A flipped image makes the model label the player's right hand as left.

    Getting this backwards is invisible in code and maddening in play: the
    blade simply follows the wrong arm.
    """
    assert KEYPOINT_NAMES[blade_keypoint("right", mirrored=True)] == "left_wrist"
    assert KEYPOINT_NAMES[blade_keypoint("left", mirrored=True)] == "right_wrist"


def test_unmirrored_right_hand_is_the_right_wrist_keypoint():
    assert KEYPOINT_NAMES[blade_keypoint("right", mirrored=False)] == "right_wrist"
    assert KEYPOINT_NAMES[blade_keypoint("left", mirrored=False)] == "left_wrist"


def test_swapping_hand_switches_the_tracked_wrist():
    game = new_game()
    assert game.hand == "right"
    game.swap_hand()
    assert game.hand == "left"


# --- slice detection ---


def test_fast_swipe_cannot_tunnel_through_fruit():
    """The whole reason slicing tests a swept segment.

    At 30 Hz a quick hand jumps ~50px per frame; testing only the current
    point would miss most cuts, and would miss hardest on the fastest swings.
    """
    blade = Blade(hit_radius=0.0)
    blade.update(np.float32([250, 300]), 0.0)
    blade.update(np.float32([550, 300]), DT)  # fruit sits between the samples
    assert blade.hits(np.float32([400, 300]), 55) is not None


def test_swipe_misses_fruit_off_the_path():
    blade = Blade()
    blade.update(np.float32([250, 300]), 0.0)
    blade.update(np.float32([550, 300]), DT)
    assert blade.hits(np.float32([400, 600]), 55) is None


def test_slow_hand_does_not_slice():
    """Resting a hand on fruit must not dissolve the board."""
    blade = Blade()
    blade.update(np.float32([400, 300]), 0.0)
    blade.update(np.float32([402, 301]), DT)
    assert not blade.slicing
    assert blade.hits(np.float32([400, 300]), 55) is None


def test_slice_direction_follows_the_swing():
    blade = Blade()
    blade.update(np.float32([250, 300]), 0.0)
    blade.update(np.float32([550, 300]), DT)
    d = blade.hits(np.float32([400, 300]), 55)
    assert d is not None
    assert d[0] == pytest.approx(1.0, abs=1e-5)  # travelling +x
    assert d[1] == pytest.approx(0.0, abs=1e-5)


def test_losing_the_hand_breaks_the_trail():
    """Otherwise the trail draws a long line across the frame on reacquire."""
    blade = Blade()
    blade.update(np.float32([100, 300]), 0.0)
    blade.update(np.float32([200, 300]), DT)
    blade.update(None, 2 * DT)
    assert blade.trail(2 * DT) == []
    assert not blade.active and not blade.slicing


def test_low_confidence_wrist_is_not_tracked():
    game = new_game()
    game.update(ControlState(present=True, pose=hand_pose(400, 300, conf=0.1)), DT)
    assert not game.blade.active


# --- flight ---


def test_fruit_peaks_inside_the_play_area():
    """Launch speed is solved from a target apex; verify it lands in frame."""
    spawner = Spawner((W, H), seed=3)
    fruits: list[Fruit] = []
    art = stub_art().fruits
    for _ in range(40):
        spawner.timer = 0.0
        spawner.update(DT, fruits, art, level=0)
    assert fruits
    for f in fruits[:25]:
        peak = f.pos[1] - (f.vel[1] ** 2) / (2 * 1500.0)
        assert 0 < peak < H * 0.9, f"apex {peak:.0f} outside the screen"


# --- rules ---


def test_slicing_scores_and_spawns_halves():
    game = new_game()
    game.fruits.append(Fruit(pos=np.float32([640, 360]), vel=np.float32([0, 0]),
                             radius=52, art=game.art.fruits[0]))
    swipe(game, 200, 1000, 360)
    assert game.score == 1
    assert len(game.fx.halves) == 2
    assert game.fx.splats


def test_dropping_fruit_costs_a_life():
    game = new_game()
    for _ in range(300):
        game.update(ControlState(present=False), DT)
    assert game.lives < game.rules.lives


def test_dropping_a_bomb_is_not_punished():
    """A bomb you let fall is a bomb you correctly avoided."""
    game = new_game(rules=Rules(lives=3))
    game.fruits.append(Fruit(pos=np.float32([640, float(H)]), vel=np.float32([0, 600]),
                             radius=52, is_bomb=True))
    for _ in range(12):
        game.update(ControlState(present=False), DT)
    assert game.lives == 3
    assert game.phase is Phase.PLAYING


def test_slicing_a_bomb_ends_the_run():
    game = new_game()
    game.fruits.append(Fruit(pos=np.float32([640, 360]), vel=np.float32([0, 0]),
                             radius=52, is_bomb=True))
    swipe(game, 200, 1000, 360)
    assert game.phase is Phase.GAME_OVER
    assert "TRASH" in game.death_reason


def test_one_swing_through_several_fruit_pays_a_combo():
    game = new_game()
    for i in range(4):
        game.fruits.append(Fruit(pos=np.float32([400 + i * 100, 360]),
                                 vel=np.float32([0, 0]), radius=52,
                                 art=game.art.fruits[0]))
    swipe(game, 100, 1100, 360)
    assert game.score == 4  # bonus not paid until the swing ends
    for _ in range(20):
        game.update(ControlState(present=False), DT)
    assert game.score > 4
    assert game.fx.banner.count == 4
    assert game.best_combo == 4


def test_two_fruit_is_not_a_combo():
    game = new_game()
    for i in range(2):
        game.fruits.append(Fruit(pos=np.float32([500 + i * 100, 360]),
                                 vel=np.float32([0, 0]), radius=52,
                                 art=game.art.fruits[0]))
    swipe(game, 100, 1100, 360)
    for _ in range(20):
        game.update(ControlState(present=False), DT)
    assert game.score == 2
    assert game.fx.banner.count == 0


def test_game_over_is_terminal_until_reset():
    game = new_game()
    game.fruits.append(Fruit(pos=np.float32([640, 360]), vel=np.float32([0, 0]),
                             radius=52, is_bomb=True))
    swipe(game, 200, 1000, 360)
    assert game.phase is Phase.GAME_OVER
    for _ in range(60):
        game.update(ControlState(present=False), DT)
    assert game.phase is Phase.GAME_OVER
    game.reset()
    assert game.phase is Phase.PLAYING
    assert game.score == 0 and game.lives == game.rules.lives


def test_no_spawning_after_game_over():
    game = new_game()
    game.phase = Phase.GAME_OVER
    before = len(game.fruits)
    for _ in range(90):
        game.update(ControlState(present=False), DT)
    assert len(game.fruits) <= before


def test_stalled_frame_does_not_fling_everything_offscreen():
    game = new_game()
    game.fruits.append(Fruit(pos=np.float32([640, 360]), vel=np.float32([0, -800]),
                             radius=52, art=game.art.fruits[0]))
    game.update(ControlState(present=False), 5.0)  # absurd stall
    assert abs(game.fruits[0].pos[1] - 360) < 400 if game.fruits else True


def test_continuous_slicing_still_pays_a_combo():
    """A player who never stops swinging must still be paid.

    The bonus resolves on a gap between slices; without a maximum swing length
    an unbroken flurry would keep the swing open forever and never pay out.
    """
    game = new_game()
    game.spawner.timer = 1e9  # only the fruit placed here
    for step in range(8):
        game.fruits.append(Fruit(pos=np.float32([640, 360]), vel=np.float32([0, 0]),
                                 radius=52, art=game.art.fruits[0]))
        swipe(game, 520, 760, 360)  # 3 frames
        for _ in range(3):  # cuts ~0.2 s apart: well inside the 0.4 s combo window
            game.update(ControlState(present=True, pose=hand_pose(760, 360)), DT)
    assert game.sliced_total == 8
    assert game.best_combo >= game.rules.combo_min, "an unbroken flurry never paid out"


def test_empty_manifest_fails_loudly():
    game = new_game()
    game.art.fruits.clear()
    with pytest.raises(RuntimeError, match="generate_fruitninja_snacks"):
        for _ in range(120):
            game.update(ControlState(present=False), DT)


def test_blit_alpha_is_clamped():
    """The uint16 blend relies on alpha <= 255 or (255 - a) wraps around."""
    from kpapp.game.fruitninja.render import FruitNinjaRenderer

    art = stub_art()
    art.background = np.zeros((H, W, 3), np.uint8)
    r = FruitNinjaRenderer((W, H), art)
    frame = np.full((H, W, 3), 100, np.uint8)
    sprite = np.zeros((40, 40, 4), np.uint8)
    sprite[:, :, :3] = 200
    sprite[:, :, 3] = 255
    r._blit(frame, sprite, (640, 360), alpha=4.0)  # absurd gain
    assert frame.min() >= 0 and frame.max() <= 255
    assert frame[360, 640].tolist() == [200, 200, 200]



# --- easier hitting ---


def _blade_through(y: float, **kw) -> Blade:
    """A fast horizontal swing at height y."""
    b = Blade(**kw)
    for i, x in enumerate(range(200, 1001, 100)):
        b.update(np.float32([x, y]), i * DT)
    return b


def test_blade_width_reaches_fruit_beside_the_path():
    """The blade is a thick stroke: a near miss by the hand still cuts."""
    fruit, r = np.float32([600, 380]), 50.0  # 80px off the hand's path
    assert _blade_through(300, hit_radius=0.0).hits(fruit, r) is None
    assert _blade_through(300, hit_radius=34.0).hits(fruit, r) is not None


def test_trail_cuts_fruit_that_arrives_after_the_swing():
    """A fruit flying into the streak just after the hand passed is cut."""
    b = _blade_through(300, hit_radius=0.0)  # isolate the trail from blade width
    late = np.float32([750, 300])  # on the 700->800 segment, swept ~0.07s ago
    # The newest segment (900->1000) is 150px away: only the trail can cut it.
    assert b.hits(late, 10.0) is not None


def test_old_trail_goes_blunt():
    """Only the recent streak cuts; an old one must not."""
    b = _blade_through(300, cut_window=0.15)
    t_end = 8 * DT
    for k in range(1, 12):  # hold still at the end for ~0.37s
        b.update(np.float32([1000, 300]), t_end + k * DT)
    assert b.hits(np.float32([400, 300]), 50) is None


def test_stepped_tracking_still_counts_as_a_swing():
    """Inference updates in steps while the game runs at 60Hz.

    Frame-to-frame speed alternates between zero and huge; the windowed speed
    must still read a real swing as one, or cuts drop out at random.
    """
    b = Blade()
    t, x = 0.0, 200.0
    for _ in range(6):
        for _ in range(3):  # three frames with no new pose
            t += 1 / 60
            b.update(np.float32([x, 300]), t)
        x += 120.0
        t += 1 / 60
        b.update(np.float32([x, 300]), t)  # then a step
        assert b.slicing, f"swing at ~1800px/s read as idle (speed {b.speed:.0f})"


def test_tracking_glitch_does_not_cut():
    """A jump across the frame is the tracker losing the hand, not a swing.

    Recorded play showed one as a stroke the full height of the frame; as a
    cut it would slice everything on that line.
    """
    b = Blade(max_jump=450)
    b.update(np.float32([100, 100]), 0.0)
    b.update(np.float32([110, 105]), DT)
    b.update(np.float32([1100, 650]), 2 * DT)  # ~1100px in one sample
    assert b.hits(np.float32([600, 380]), 60) is None
    assert len(b.trail(2 * DT)) == 1, "the glitch should start a fresh stroke"


def test_opening_waves_are_gentle():
    """Recorded rounds ended in ~3 seconds: openings must be small, bomb-free."""
    spawner = Spawner((W, H), seed=5)
    art = stub_art().fruits
    for wave in range(1, 7):
        fruits: list[Fruit] = []
        spawner.timer = 0.0
        spawner.update(DT, fruits, art, level=0)
        assert 1 <= len(fruits) <= 2, f"wave {wave} launched {len(fruits)}"
        if wave <= 4:
            assert not any(f.is_bomb for f in fruits), f"bomb in opening wave {wave}"


def test_resume_does_not_read_as_a_giant_swing():
    """The hand moves during a pause while the game clock stands still."""
    game = new_game()
    game.update(ControlState(present=True, pose=hand_pose(200, 300)), DT)
    game.on_resume()
    assert not game.blade.active


# --- variety, fairness, feedback ---


def _wave_signature(seed=None, waves=6):
    sp = Spawner((W, H), seed=seed)
    out = []
    for _ in range(waves):
        fruits: list[Fruit] = []
        sp.timer = 0.0
        sp.update(DT, fruits, stub_art().fruits, level=3)
        out.append(tuple((round(float(f.pos[0])), f.art.name if f.art else "bomb") for f in fruits))
    return out


def test_every_game_is_different():
    """The spawner used to run on a fixed seed: every round was identical."""
    runs = {tuple(_wave_signature()) for _ in range(5)}
    assert len(runs) == 5, "unseeded games repeated the same fruit"


def test_a_seed_still_reproduces_a_game():
    assert _wave_signature(seed=42) == _wave_signature(seed=42)


def test_bombs_never_crowd_a_fruit():
    """Every fruit must be cuttable without touching a bomb.

    Checked independently of the spawner's own projection: real game physics,
    stepped frame by frame for 90 s at the hardest difficulty, measuring every
    bomb-fruit distance while both are on screen.
    """
    sp = Spawner((W, H), seed=7, clearance=34.0)
    art = stub_art().fruits
    bodies: list[Fruit] = []
    worst = float("inf")
    dt = 1 / 60
    for _ in range(int(90 / dt)):
        sp.update(dt, bodies, art, level=12)
        for b in bodies:
            b.step(dt)
        bodies = [b for b in bodies if b.pos[1] < H + 110]
        on = [b for b in bodies if -40 < b.pos[0] < W + 40 and -60 < b.pos[1] < H + 10]
        for bomb in (b for b in on if b.is_bomb):
            for fruit in (f for f in on if not f.is_bomb):
                gap = float(np.linalg.norm(bomb.pos - fruit.pos)) - sp.separation(bomb, fruit)
                worst = min(worst, gap)
    bombs_thrown = sp.wave  # sanity: the run did produce waves
    assert bombs_thrown > 40
    assert worst >= -2.0, f"a bomb came {-worst:.0f}px inside the safe distance of a fruit"


def test_bombs_still_appear_at_high_difficulty():
    """Safety must not quietly remove bombs from the game."""
    sp = Spawner((W, H), seed=3)
    art = stub_art().fruits
    bodies: list[Fruit] = []
    thrown = 0
    for _ in range(60):
        before = sum(b.is_bomb for b in bodies)
        sp.timer = 0.0
        sp.update(DT, bodies, art, level=12)
        thrown += sum(b.is_bomb for b in bodies) - before
        bodies = []  # judge each wave on its own
    assert thrown >= 8, f"only {thrown} bombs in 60 hard waves"


def test_cutting_fruit_pops_up_its_points():
    game = new_game()
    game.spawner.timer = 1e9
    two = FruitArt("pineapple", *(game.art.fruits[0].whole,) * 3, (60, 200, 240), 52, 2)
    game.fruits.append(Fruit(pos=np.float32([500, 360]), vel=np.float32([0, 0]), radius=52,
                             art=game.art.fruits[0]))
    game.fruits.append(Fruit(pos=np.float32([800, 360]), vel=np.float32([0, 0]), radius=52, art=two))
    swipe(game, 300, 1000, 360)
    texts = sorted(p.text for p in game.fx.popups)
    assert texts == ["+1", "+2"]
    assert next(p for p in game.fx.popups if p.text == "+2").big
    assert game.fx.score_bump > 0.5, "the score counter should pulse"


def test_popups_rise_and_expire():
    game = new_game()
    game.spawner.timer = 1e9
    game.fruits.append(Fruit(pos=np.float32([640, 360]), vel=np.float32([0, 0]), radius=52,
                             art=game.art.fruits[0]))
    swipe(game, 400, 900, 360)
    pop = game.fx.popups[0]
    y0 = float(pop.pos[1])
    for _ in range(10):
        game.update(ControlState(present=False), DT)
    assert pop.pos[1] < y0, "popup should float upward"
    for _ in range(40):
        game.update(ControlState(present=False), DT)
    assert not game.fx.popups, "popups should be gone after about a second"


def test_popups_render_without_error_near_edges():
    from kpapp.game.fruitninja.entities import Popup
    from kpapp.game.fruitninja.render import FruitNinjaRenderer

    game = new_game()
    for pos in ([5, 5], [W - 3, H - 3], [640, 360], [-50, 100]):
        game.fx.popups.append(Popup(np.float32(pos), "+2", (60, 60, 220), big=True, life=0.5))
    frame = np.zeros((H, W, 3), np.uint8)
    FruitNinjaRenderer((W, H), game.art)._draw_popups(frame, game)



# --- the raccoon ---


def test_a_bite_opens_the_raccoons_mouth_briefly():
    game = new_game()
    game.spawner.timer = 1e9
    game.fruits.append(Fruit(pos=np.float32([640, 360]), vel=np.float32([0, 0]), radius=52,
                             art=game.art.fruits[0]))
    swipe(game, 400, 900, 360)
    assert game.fx.chomp > 0.5, "eating should show the open-mouth frame"
    for _ in range(12):
        game.update(ControlState(present=False), DT)
    assert game.fx.chomp == 0.0, "the mouth closes again after ~0.2 s"


def test_raccoon_is_drawn_on_the_palm():
    """The cursor art replaces the plain ring when it is available."""
    from kpapp.game.fruitninja.render import FruitNinjaRenderer

    art = stub_art()
    face = np.zeros((120, 120, 4), np.uint8)
    face[30:90, 30:90] = (200, 60, 120, 255)  # a purple square standing in for the raccoon
    art.cursor_idle = face
    game = new_game()
    game.art = art
    game.spawner.timer = 1e9
    game.update(ControlState(present=True, pose=hand_pose(640, 360)), DT)
    frame = np.zeros((H, W, 3), np.uint8)
    r = FruitNinjaRenderer((W, H), art)
    r._draw_cursor(frame, game, np.float32([640, 360]))
    assert tuple(frame[360, 640]) == (200, 60, 120), "raccoon sprite should sit on the palm"


def test_hit_area_matches_the_raccoon_head():
    assert new_game().blade.hit_radius == 44.0
