"""The drawing pieces the Thaumaturge screens share: colours, a curses canvas and a plain-text one, boxes, lines with
their keys marked, and starting inside tmux.

A screen draws on a canvas (`cv.put(y, x, text, fg, bg)`), never on curses directly, so the same drawing can be
printed as text (--once) and tested without a terminal.

Colours can be set in ~/.config/thaumaturge/colours.json, e.g. {"key": "#00ff00", "muted": "#9b8f7f"}: any of COL's
names, as #rrggbb.
"""

from __future__ import annotations

import curses
import json
import os
import shutil
import sys
import unicodedata
from pathlib import Path

COL = {"frame": "#7a6e60", "text": "#e6dccd", "muted": "#9b8f7f", "ok": "#7fb276", "warn": "#d9b25f",
       "bad": "#e07a68", "attn": "#4fa3e0", "ff": "#a3be8c", "mem": "#b48ead", "cyan": "#88c0d0",
       "cold": "#e8c547", "hot": "#d9452b", "sel": "#3a332c", "dim": "#4a4038", "key": "#00ff00"}


def config_dir() -> Path:
    """~/.config/thaumaturge (for whoever runs it with sudo, theirs; XDG_CONFIG_HOME if set)."""
    user = os.environ.get("SUDO_USER")
    if user:
        return Path(os.path.expanduser(f"~{user}")) / ".config" / "thaumaturge"
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "thaumaturge"


def load_colours(path: Path | None = None) -> list[str]:
    """COL updated from the colours file, if there is one; what in it was left out, and why."""
    path = path or config_dir() / "colours.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return []
    except ValueError as e:
        return [f"{path}: not JSON as written ({e})"]
    left = []
    for k, v in (saved.items() if isinstance(saved, dict) else []):
        if k in COL and isinstance(v, str) and len(v) == 7 and v.startswith("#") \
                and all(c in "0123456789abcdefABCDEF" for c in v[1:]):
            COL[k] = v.lower()
        else:
            left.append(f"{path}: {k} left as it was (names: {', '.join(sorted(COL))}; colours as #rrggbb)")
    return left


# -- canvases: curses, or plain text for --once and the tests

class TextCanvas:
    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.cells = [[(" ", None, None) for _ in range(w)] for _ in range(h)]

    def put(self, y: int, x: int, s: str, fg: str | None = None, bg: str | None = None) -> None:
        if not 0 <= y < self.h:
            return
        for ch in s:
            if 0 <= x < self.w:
                self.cells[y][x] = (ch, fg, bg)
            if unicodedata.east_asian_width(ch) in ("W", "F"):        # a wide character covers the next cell too
                x += 1
                if 0 <= x < self.w:
                    self.cells[y][x] = ("", fg, bg)
            x += 1

    def text(self, colour: bool = False) -> str:
        out = []
        for row in self.cells:
            if not colour:
                out.append("".join(c for c, _, _ in row).rstrip())
                continue
            line = ""
            for c, fg, bg in row:
                esc = ""
                if fg:
                    r, g, b = (int(fg[i:i + 2], 16) for i in (1, 3, 5))
                    esc += f"\x1b[38;2;{r};{g};{b}m"
                if bg:
                    r, g, b = (int(bg[i:i + 2], 16) for i in (1, 3, 5))
                    esc += f"\x1b[48;2;{r};{g};{b}m"
                line += esc + c + ("\x1b[0m" if esc else "")
            out.append(line.rstrip())
        while out and not out[-1].strip():
            out.pop()
        return "\n".join(out)


class Colours:
    """Hex colours as curses colour pairs: the nearest of the terminal's 256 (or 8)."""

    def __init__(self):
        self.pairs: dict[tuple, int] = {}
        self.colours = curses.COLORS if curses.has_colors() else 0

    def number(self, hexcode: str) -> int:
        try:
            r, g, b = (int(hexcode.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return -1
        if self.colours >= 256:
            cube = [0, 95, 135, 175, 215, 255]

            def near(v):
                return min(range(6), key=lambda i: abs(cube[i] - v))
            return 16 + 36 * near(r) + 6 * near(g) + near(b)
        if self.colours >= 8:
            return (1 if r > 127 else 0) + (2 if g > 127 else 0) + (4 if b > 127 else 0)
        return -1

    def __call__(self, fg: str, bold: bool = False) -> int:
        num = self.number(fg)
        key = (num, bold)
        if key not in self.pairs:
            n = len(self.pairs) + 1
            if self.colours and n < curses.COLOR_PAIRS:
                curses.init_pair(n, num, -1)
                self.pairs[key] = curses.color_pair(n) | (curses.A_BOLD if bold else 0)
            else:
                self.pairs[key] = curses.A_BOLD if bold else 0
        return self.pairs[key]


class CursesCanvas:
    def __init__(self, win):
        self.win = win
        self.c = Colours()
        self.bgpairs: dict = {}
        self.h, self.w = win.getmaxyx()

    def attr(self, fg, bg):
        if not bg:
            return self.c(fg or COL["text"])
        key = (fg, bg)
        if key not in self.bgpairs:
            n = len(self.c.pairs) + len(self.bgpairs) + 1
            if self.c.colours and n < curses.COLOR_PAIRS:
                curses.init_pair(n, self.c.number(fg or COL["text"]), self.c.number(bg))
                self.bgpairs[key] = curses.color_pair(n)
            else:
                self.bgpairs[key] = curses.A_REVERSE
        return self.bgpairs[key]

    def put(self, y: int, x: int, s: str, fg: str | None = None, bg: str | None = None) -> None:
        if not (0 <= y < self.h) or x >= self.w:
            return
        if x < 0:
            s, x = s[-x:], 0
        try:
            self.win.addnstr(y, x, s, max(0, self.w - x - (1 if y == self.h - 1 else 0)), self.attr(fg, bg))
        except curses.error:
            pass


# -- drawing pieces

def width(s: str) -> int:
    """How many terminal columns `s` takes: Chinese, Japanese and Korean take two, marks stacked on a letter none."""
    return sum(0 if unicodedata.combining(c) or unicodedata.category(c) in ("Mn", "Me", "Cf")
               else 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


KEY_OPEN, KEY_CLOSE = "‹", "›"        # in a line of text, ‹s› marks a key: drawn in COL["key"], the marks left out


def plain(s: str) -> str:
    """The line as it shows, without the key marks."""
    return s.replace(KEY_OPEN, "").replace(KEY_CLOSE, "")


def keyed(cv, y: int, x: int, s: str, fg: str | None = None, bg: str | None = None, width: int | None = None) -> None:
    """A line with its keys (‹s›) in the key colour, the rest in fg."""
    room = len(plain(s)) if width is None else max(0, width)
    on = False
    for part in s.replace(KEY_CLOSE, KEY_OPEN).split(KEY_OPEN):
        if room <= 0:
            break
        part = part[:room]
        if part:
            cv.put(y, x, part, COL["key"] if on else fg, bg)
        x += len(part)
        room -= len(part)
        on = not on


def box(cv, y: int, x: int, h: int, w: int, title: str) -> None:
    n = width(plain(title))
    cv.put(y, x, "╭─ " + " " * n + " " + "─" * max(0, w - n - 5) + "╮", COL["frame"])
    keyed(cv, y, x + 3, title, COL["text"])
    for i in range(1, h - 1):
        cv.put(y + i, x, "│", COL["frame"])
        cv.put(y + i, x + w - 1, "│", COL["frame"])
    cv.put(y + h - 1, x, "╰" + "─" * (w - 2) + "╯", COL["frame"])


def into_tmux(session: str, argv: list[str]) -> None:
    """Start the screen inside tmux, so a dropped ssh connection leaves it running: the same command again
    reattaches to it (tmux's Ctrl-b d leaves it on purpose). Not when already inside tmux, with no terminal, or where
    tmux isn't installed. `argv` is the whole command to run inside."""
    if os.environ.get("TMUX") or not (sys.stdin.isatty() and sys.stdout.isatty()) or not shutil.which("tmux"):
        return
    session = "".join(c if c.isalnum() or c in "-_" else "-" for c in session)[:60]
    os.execvp("tmux", ["tmux", "new-session", "-A", "-s", session, *argv])
