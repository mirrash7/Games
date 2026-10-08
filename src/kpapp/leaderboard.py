"""High-score tables, one per game, with arcade-style names of up to three
characters (port of web/js/leaderboard.js).

Two backends with the same surface: begin, top and submit. Each returns a
`concurrent.futures.Future`, so the 60 Hz render loop never waits on disk or
network; the shell polls `.done()`.

  LocalBoard   this computer only (a JSON file). Always works.
  ServerBoard  the worldwide table: the Cloudflare Worker in leaderboard/,
               shared with the browser build (docs/LEADERBOARD.md).

`create_board()` picks one. The remote one falls back to the local table
whenever the server can't be reached or refuses a score, so a flaky
connection never blocks the game.

`begin(game)` is called when a round starts: the server records the time and
later refuses scores the game couldn't produce in that time.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

NAME_MAX = 3
BOARD_SIZE = 10
MAX_SCORE = 100_000  # matches the server's limit; anything above is not a real run

# Three letters can still spell abuse on a public board. A short, obvious list
# (the same as the site's and the server's); the player is asked for another name.
BLOCKED = {
    "ASS", "CUM", "FUK", "FUC", "FCK", "FKU", "DIK", "DIC", "COK", "KKK", "NIG", "NGR", "FAG", "TIT",
    "SHT", "CNT", "SEX", "XXX", "NAZ", "SS", "KYS", "PIS", "POO", "WTF",
}

DEFAULT_FILE = Path(os.environ.get("KP_LEADERBOARD_FILE", Path.home() / ".kp" / "leaderboard.json"))
SITE_CONFIG = Path(__file__).resolve().parents[2] / "web" / "js" / "config.js"


def clean_name(s: object) -> str:
    """Upper-case A-Z and 0-9 only, at most three."""
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())[:NAME_MAX]


def name_allowed(name: str) -> bool:
    n = clean_name(name)
    return bool(n) and n == name and n not in BLOCKED


def sort_entries(entries: list[dict]) -> list[dict]:
    """Highest score first; on a tie, whoever got there first."""
    return sorted(entries, key=lambda e: (-e["score"], e["at"]))


def qualifies(entries: list[dict], score: float, size: int = BOARD_SIZE) -> bool:
    """Would `score` earn a place on a board currently holding `entries`?"""
    if not score > 0 or score > MAX_SCORE:
        return False
    if len(entries) < size:
        return True
    return score > sort_entries(entries)[size - 1]["score"]  # ties go to the earlier run


def _done(value) -> Future:
    f: Future = Future()
    f.set_result(value)
    return f


class LocalBoard:
    """This computer's own table, in a small JSON file."""

    scope = "device"

    def __init__(self, path: Path | str | None = DEFAULT_FILE) -> None:
        self.path = Path(path) if path is not None else None
        self._memory: dict[str, list[dict]] = {}  # used if the file can't be written
        self._lock = threading.Lock()

    def _all(self) -> dict[str, list[dict]]:
        if self.path is None:
            return self._memory
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return dict(self._memory)

    def begin(self, game: str) -> Future:
        return _done(None)  # nothing to verify locally

    def top(self, game: str) -> Future:
        return _done(sort_entries(self._all().get(game, []))[:BOARD_SIZE])

    def submit(self, game: str, name: str, score: int, session: str | None = None) -> Future:
        with self._lock:
            data = self._all()
            entry = {"id": f"{int(time.time() * 1000)}-{secrets.token_hex(3)}",
                     "name": clean_name(name), "score": int(score), "at": int(time.time() * 1000)}
            data[game] = sort_entries([*data.get(game, []), entry])[:BOARD_SIZE]
            self._memory = data
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = self.path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(data, indent=1))
                    tmp.replace(self.path)  # never leave a half-written file
                except OSError:
                    pass  # keep the in-memory copy for this session
            return _done((data[game], entry["id"]))


class LeaderboardError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class ServerBoard:
    """The worldwide table: the leaderboard Worker (leaderboard/src/worker.js)."""

    scope = "world"

    def __init__(self, url: str, timeout: float = 6.0, opener=urllib.request.urlopen) -> None:
        self.base = url.rstrip("/")
        self.timeout = timeout
        self._open = opener
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="leaderboard")

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with self._open(req, timeout=self.timeout) as res:
                return json.loads(res.read() or b"{}")
        except urllib.error.HTTPError as err:
            try:
                message = json.loads(err.read() or b"{}").get("error", f"leaderboard {err.code}")
            except ValueError:
                message = f"leaderboard {err.code}"
            raise LeaderboardError(err.code, message) from None

    def begin(self, game: str) -> Future:
        return self._pool.submit(lambda: self._call("POST", "/sessions", {"game": game})["session"])

    def top(self, game: str) -> Future:
        q = urllib.parse.urlencode({"game": game})
        return self._pool.submit(lambda: self._call("GET", f"/scores?{q}")["entries"])

    def submit(self, game: str, name: str, score: int, session: str | None = None) -> Future:
        def run():
            out = self._call("POST", "/scores", {"game": game, "name": clean_name(name),
                                                  "score": int(round(score)), "session": session})
            return out["entries"], out["id"]
        return self._pool.submit(run)


class FallbackBoard:
    """The worldwide board, falling back to this computer's on any failure.

    `scope` says which one the last answer came from ("world" or "device").
    """

    def __init__(self, remote: ServerBoard, local: LocalBoard) -> None:
        self.remote, self.local = remote, local
        self.scope = "world"
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="leaderboard-fallback")

    def _try(self, remote_call, local_call, what: str) -> Future:
        def run():
            try:
                value = remote_call().result()
                self.scope = "world"
                return value
            except Exception as exc:  # noqa: BLE001 - any failure means "use this computer's"
                print(f"[kp] worldwide leaderboard {what}: {exc}; using this computer's")
                self.scope = "device"
                return local_call().result()
        return self._pool.submit(run)

    def begin(self, game: str) -> Future:
        return self._try(lambda: self.remote.begin(game), lambda: _done(None), "unavailable for this round")

    def top(self, game: str) -> Future:
        return self._try(lambda: self.remote.top(game), lambda: self.local.top(game), "unavailable")

    def submit(self, game: str, name: str, score: int, session: str | None = None) -> Future:
        if not session:  # the round began while the server was unreachable
            self.scope = "device"
            return self.local.submit(game, name, score)
        return self._try(lambda: self.remote.submit(game, name, score, session),
                         lambda: self.local.submit(game, name, score), "did not take the score")


def site_url(config: Path = SITE_CONFIG) -> str:
    """The worldwide server the website uses (web/js/config.js), if set."""
    try:
        m = re.search(r'url:\s*"([^"]*)"', config.read_text())
    except OSError:
        return ""
    return m.group(1) if m else ""


def create_board(url: str | None = None, local: LocalBoard | None = None):
    """`url`: "local" for this computer only, a server URL, or None to use the
    same server as the website (web/js/config.js) when one is configured."""
    local = local or LocalBoard()
    if url is None:
        url = site_url()
    if not url or url == "local":
        return local
    return FallbackBoard(ServerBoard(url), local)
