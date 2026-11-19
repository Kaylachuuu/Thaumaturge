"""The BitScryer: a text-mode workbench for learning to program the card, and for going on programming it after
The design: docs/bitscryer.md.

    bitscryer                      the lessons (it opens the one you were on)
    bitscryer ~/kernels/mine       a project of your own: a folder with a .cu or .ptx file (and check.py, if any)
    bitscryer new ~/kernels/mine   start a project: a kernel and a check.py to change
    bitscryer --selftest           every lesson's answer built and run on the card: does BitScryer work here?
    bitscryer --once               one screen as plain text, no keys

You type your kernel right in the left pane (BitScryer's own editor, thaumaturge/common/editor.py). Each save (Ctrl-S) builds
it again, shows the PTX and the card's own instructions (SASS) beside it, and (if there's a check.py) runs it on the
card, checks it against the CPU, and times it against the card's top speed. The line the cursor is on lights up the
PTX and SASS it became.

Keys: the editor's own (arrows, Shift to select, Ctrl-S save, Ctrl-Z undo, Ctrl-Y redo, Ctrl-C/X/V copy, cut and
paste, Ctrl-K delete a line, Ctrl-D duplicate it, Ctrl-F find, Ctrl-G go to a line, Tab indent), and for BitScryer:
F1 the lesson · F2 a hint · F3 how the work splits into blocks and warps · F4 choose the card · F5 run again ·
F6 the lessons · F7 the build's full output · F8 PTX or SASS · F9 these keys · F10 or Ctrl-Q quit (saving first).
Where F-keys don't come through, Esc then a letter does the same: l h w c r n o t ? q (and e: your own editor, nano
or vim, in a tmux pane beside it, if you'd rather).

It runs only on the cards cards.py allows (see settings.py), and never as root.
"""

from __future__ import annotations

import curses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thaumaturge.common.screen import COL, CursesCanvas, TextCanvas, box, keyed, load_colours  # noqa: E402
from thaumaturge.bitscryer import build as builder  # noqa: E402
from thaumaturge.bitscryer import cards as cardlib  # noqa: E402
from thaumaturge.bitscryer import lessons as lessonlib  # noqa: E402
from thaumaturge.common.editor import Editor  # noqa: E402
from thaumaturge.bitscryer.picture import picture  # noqa: E402

CODE = Path(__file__).resolve().parents[2]          # the repository: thaumaturge/ is in it
RUN_LIMIT_S = 30.0                    # a run (test data, the checked run and the timing) is stopped after this

PLAIN_ERRORS = {
    "ILLEGAL_ADDRESS": "the kernel read or wrote memory outside its arrays (an index past the end?)",
    "MISALIGNED_ADDRESS": "a wide load or store (float4, uint4) at an address that isn't a multiple of its size",
    "LAUNCH_OUT_OF_RESOURCES": "the block needs more registers than the card has for it: try a smaller block",
    "INVALID_VALUE": "the launch's numbers aren't allowed (a block too big, or a grid of 0?)",
    "ILLEGAL_INSTRUCTION": "an instruction this card can't run (hand-written PTX for a newer card?)",
    "INVALID_PTX": "the PTX doesn't suit this card",
    "NO_BINARY_FOR_GPU": "it was built for a different card",
    "LAUNCH_TIMEOUT": "the kernel ran too long",
    "ASSERT": "an assert() in the kernel failed",
}


def explain(err: str) -> str:
    for k, v in PLAIN_ERRORS.items():
        if k in err:
            return v
    return ""


def fmt_ms(ms: float) -> str:
    return f"{ms * 1000:.1f} µs" if ms < 0.1 else f"{ms:.3f} ms" if ms < 10 else f"{ms:.1f} ms"


def fmt_rate(x: float, unit: str) -> str:
    return f"{x / 1000:.2f} T{unit}" if x >= 1000 else f"{x:.1f} G{unit}"


def wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, max(10, width), break_long_words=True, break_on_hyphens=False) or [""]


def markdown_lines(text: str, width: int) -> list[tuple[str, str]]:
    """Lesson text as coloured lines: headings, code, and the rest wrapped to the width."""
    out: list[tuple[str, str]] = []
    code = False
    para: list[str] = []

    def flush():
        if para:
            joined = " ".join(s.strip() for s in para)
            lead = "  " if para[0].lstrip().startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.",
                                                        "9.")) else ""
            for i, w in enumerate(wrap(joined, width - len(lead))):
                out.append(((lead if i else "") + w, "text"))
            para.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            flush()
            code = not code
            continue
        if code:
            out.append(("  " + line, "ff"))
            continue
        if line.startswith("#"):
            flush()
            out.append((line.lstrip("#").strip(), "attn"))
            continue
        if not line.strip():
            flush()
            out.append(("", "text"))
            continue
        if line.lstrip().startswith(("- ", "* ")) or line[:3].rstrip(".").isdigit():
            flush()
        para.append(line)
    flush()
    return out


class App:
    def __init__(self, work: lessonlib.Work | None = None, progress: lessonlib.Progress | None = None,
                 card_pick: str | None = None):
        self.progress = progress or lessonlib.Progress()
        self.lessons = lessonlib.lessons()
        self.work = work or self._last_lesson()
        if self.work.lesson:
            lessonlib.open_lesson(self.work)
        self.editor = Editor(self.work.source)
        self.cursor_at: tuple[int, int] | None = None
        self.cards: list[cardlib.Card] = []
        self.world = cardlib.World()
        self.card: cardlib.Card | None = None
        self.info: dict = {}
        self.built: builder.Built | None = None
        self.result: dict | None = None
        self.status = "starting"
        self.busy = False
        self.right = "sass"
        self.overlay: str | None = None
        self.ov_scroll = 0
        self.ov_sel = 0
        self.message = ""
        self.mtime = 0.0
        self.lock = threading.Lock()
        self.pending = False
        self.refresh_cards()
        self.choose_card(card_pick or self.progress.get("card"))

    # -- what is open, and on which card

    def _last_lesson(self) -> lessonlib.Work:
        last = self.progress.get("last")
        for w in self.lessons:
            if w.key == last:
                return w
        for w in self.lessons:
            if not self.progress.passed(w.key):
                return w
        return self.lessons[0]

    def refresh_cards(self) -> None:
        self.cards = cardlib.read_cards()
        self.world = cardlib.World.read()

    def allowed(self, c: cardlib.Card) -> str:
        return cardlib.why_not(c, self.world)

    def choose_card(self, want: str | None) -> None:
        """The card asked for (a UUID or a GPU number) if it may be used; else the first that may, smallest first, so
        the T1000 before the V100."""
        pick = None
        for c in self.cards:
            if want and (c.uuid == want or str(c.index) == str(want).removeprefix("GPU")):
                pick = c
        if pick is None or cardlib.never(pick, self.world):
            ok = [c for c in self.cards if not self.allowed(c)]
            ok.sort(key=lambda c: c.mem_total)
            pick = ok[0] if ok else None
        if pick and cardlib.never(pick, self.world):
            pick = None
        self.card = pick
        self.info = (self.progress.get("cards") or {}).get(pick.uuid, {}) if pick else {}
        if pick:
            self.progress.set("card", pick.uuid)

    @property
    def cursor(self) -> int:
        """The line of your file the cursor is on (from 1): its PTX and SASS light up."""
        return self.editor.row + 1

    @cursor.setter
    def cursor(self, n: int) -> None:
        self.editor.goto(n - 1, self.editor.col)

    @property
    def sm(self) -> str:
        return self.card.sm if self.card else "sm_75"

    def open(self, work: lessonlib.Work) -> None:
        if self.editor.dirty:                            # your work is never left behind unsaved
            self.editor.save()
        self.work = work
        if work.lesson:
            lessonlib.open_lesson(work)
            self.progress.set("last", work.key)
        self.editor = Editor(work.source)
        self.built, self.result, self.mtime = None, None, 0.0
        self.overlay = None

    # -- building and running, off the screen's thread

    def poll_file(self) -> bool:
        try:
            m = self.work.source.stat().st_mtime
        except OSError:
            return False
        if m != self.mtime:
            self.mtime = m
            if not self.editor.dirty:                    # changed outside (or just saved): show what's on disk
                try:
                    text = self.work.source.read_text(errors="replace")
                except OSError:
                    text = None
                if text is not None and text != self.editor.text():
                    self.editor.load()
            return True
        return False

    def start(self) -> None:
        with self.lock:
            if self.busy:
                self.pending = True
                return
            self.busy = True
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        try:
            while True:
                self.pending = False
                self.cycle()
                if not self.pending:
                    break
        finally:
            self.busy = False

    def cycle(self) -> None:
        if not self.work.source.exists():
            self.status = f"no file yet: {self.work.source}"
            return
        self.status = "building"
        b = builder.build(self.work.source, self.sm)
        self.built = b
        if not b.ok:
            self.result = None
            self.status = f"didn't build ({b.stage}): {len(b.errors)} problem{'s' if len(b.errors) != 1 else ''}"
            return
        if not self.work.check:
            self.status = f"built in {b.seconds:.1f} s (no check.py: nothing to run)"
            return
        self.refresh_cards()
        if not self.card:
            self.status = "built; not run: no card BitScryer may use right now (c to see why)"
            return
        live = next((c for c in self.cards if c.uuid == self.card.uuid), None)
        why = "it is gone from nvidia-smi" if live is None else self.allowed(live)
        if why:
            self.status = f"built; not run on {self.card.short()}: {why}"
            return
        if "tensor_cores" in self.work.needs and not live.tensor_cores:
            self.status = f"built; not run: {live.short()} has no tensor cores (c for another card)"
            return
        self.status = f"running on {live.short()}"
        self.result = self.run(live, b)
        if self.result.get("card"):
            self.info = self.result["card"]
            saved = self.progress.get("cards") or {}
            saved[live.uuid] = self.info
            self.progress.set("cards", saved)
        parts = self.result.get("parts") or []
        if self.result.get("error"):
            self.status = "the run failed"
        elif parts and all(p.get("status") == "right" for p in parts):
            self.status = "right"
            if self.work.lesson:
                self.progress.mark_passed(self.work.key)
        else:
            self.status = "not right yet"

    def run(self, card: cardlib.Card, b: builder.Built) -> dict:
        s = builder.settings(self.work.source.read_text(errors="replace"))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(s, f)
            sfile = f.name
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"),
               "LANG": os.environ.get("LANG", "C.UTF-8"), "CUDA_VISIBLE_DEVICES": cardlib.visible_uuid(card),
               "PYTHONPATH": str(CODE), "OMP_NUM_THREADS": "4"}
        argv = cardlib.gentle([sys.executable, "-m", "thaumaturge.bitscryer.runner", "run", str(self.work.check), str(b.cubin),
                               sfile])
        t0 = time.time()
        try:
            p = subprocess.Popen(argv, env=env, cwd=str(CODE), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, start_new_session=True)
            try:
                out, err = p.communicate(timeout=RUN_LIMIT_S)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 9)
                p.communicate()
                return {"error": f"stopped after {RUN_LIMIT_S:.0f} s: a kernel that never ends? (a loop whose end "
                                 f"is never reached, or a __syncthreads() some threads never get to). Stopping the "
                                 f"program freed the card."}
        finally:
            try:
                os.unlink(sfile)
            except OSError:
                pass
        line = out.strip().splitlines()[-1] if out.strip() else ""
        try:
            res = json.loads(line)
        except ValueError:
            res = {"error": f"the runner ended without an answer (exit {p.returncode}): {(err or out)[-600:]}"}
        res["seconds"] = time.time() - t0
        return res

    # -- keys

    COMMANDS = {curses.KEY_F1: "l", curses.KEY_F2: "h", curses.KEY_F3: "w", curses.KEY_F4: "c", curses.KEY_F5: "r",
                curses.KEY_F6: "n", curses.KEY_F7: "o", curses.KEY_F8: "t", curses.KEY_F9: "?", curses.KEY_F10: "q",
                "\x11": "q", "\x12": "r"}                     # Ctrl-Q, Ctrl-R

    def key(self, k, alt: bool = False) -> bool:
        """One key, as get_wch gives it (alt: it came after Esc). False to quit."""
        self.message = ""
        cmd = k if alt and isinstance(k, str) else self.COMMANDS.get(k)
        if self.overlay:
            if cmd and cmd != {"lesson": "l", "hints": "h", "warps": "w", "cards": "c", "lessons": "n", "log": "o",
                               "help": "?"}.get(self.overlay):
                self.overlay = None                       # another command: this page closes, that one opens
            else:
                if cmd:
                    k = {"h": "h"}.get(cmd, "\x1b")      # the same command again: closes it (F2 again: next hint)
                return self.key_overlay(ord(k) if isinstance(k, str) and len(k) == 1 else k)
        if cmd:
            return self.command(cmd)
        if alt:
            return True
        if self.editor.key(k):
            self.message = self.editor.message
            return True
        return True

    def command(self, c: str) -> bool:
        if c == "q":
            if self.editor.dirty:
                self.editor.save()
            return False
        if c == "t":
            self.right = "ptx" if self.right == "sass" else "sass"
        elif c == "r":
            if self.editor.dirty:
                self.editor.save()                       # the file watcher builds it
            else:
                self.start()
        elif c == "e":
            if self.editor.dirty:
                self.editor.save()
            self.edit()
        elif c in ("l", "?", "w", "o"):
            self.overlay = {"l": "lesson", "?": "help", "w": "warps", "o": "log"}[c]
            self.ov_scroll = 0
        elif c == "h":
            hints = self.work.hints()
            seen = self.progress.hints_seen(self.work.key)
            if not hints:
                self.message = "No hints for this one."
            else:
                if seen == 0:
                    self.progress.see_hint(self.work.key, 1)
                self.overlay, self.ov_scroll = "hints", 0
        elif c == "c":
            self.refresh_cards()
            self.overlay = "cards"
            self.ov_sel = next((i for i, cd in enumerate(self.cards) if self.card and cd.uuid == self.card.uuid), 0)
        elif c == "n":
            self.overlay = "lessons"
            self.ov_sel = next((i for i, w in enumerate(self.lessons) if w.key == self.work.key), 0)
        return True

    def key_overlay(self, ch: int) -> bool:
        if ch in (27, ord("q")):
            self.overlay = None
            return True
        if self.overlay in ("cards", "lessons"):
            n = len(self.cards) if self.overlay == "cards" else len(self.lessons)
            if ch in (curses.KEY_DOWN, ord("j")):
                self.ov_sel = min(n - 1, self.ov_sel + 1)
            elif ch in (curses.KEY_UP, ord("k")):
                self.ov_sel = max(0, self.ov_sel - 1)
            elif ch in (10, 13, curses.KEY_ENTER):
                if self.overlay == "cards" and self.cards:
                    c = self.cards[self.ov_sel]
                    why = self.allowed(c)
                    if cardlib.never(c, self.world):
                        self.message = f"Not that one: {cardlib.never_why(c, self.world)}."
                        return True
                    self.card = c
                    self.info = (self.progress.get("cards") or {}).get(c.uuid, {})
                    self.progress.set("card", c.uuid)
                    self.overlay = None
                    self.message = f"Card: {c.short()}" + (f" (not free now: {why})" if why else "")
                    self.start()
                elif self.overlay == "lessons":
                    self.open(self.lessons[self.ov_sel])
                    self.poll_file()
                    self.start()
            return True
        if ch == ord("h") and self.overlay == "hints":
            hints = self.work.hints()
            seen = self.progress.hints_seen(self.work.key)
            if seen < len(hints):
                self.progress.see_hint(self.work.key, seen + 1)
            else:
                self.message = "That's all the hints."
        elif ch in (curses.KEY_DOWN, ord("j")):
            self.ov_scroll += 1
        elif ch in (curses.KEY_UP, ord("k")):
            self.ov_scroll = max(0, self.ov_scroll - 1)
        elif ch in (curses.KEY_NPAGE, ord(" ")):
            self.ov_scroll += 20
        elif ch in (curses.KEY_PPAGE, ord("b")):
            self.ov_scroll = max(0, self.ov_scroll - 20)
        return True

    def edit(self) -> None:
        """Your editor on your file: beside BitScryer in tmux, or in its place outside tmux."""
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
        path = str(self.work.source)
        if os.environ.get("TMUX") and shutil.which("tmux"):
            subprocess.run(["tmux", "split-window", "-h", "-l", "50%", f"{editor} '{path}'"],
                           capture_output=True)
            self.message = "Your editor is open on the right (Ctrl-b then an arrow moves between them). Save to run."
            return
        curses.endwin()
        subprocess.run([*editor.split(), path])
        curses.doupdate()

    # -- drawing

    def source_lines(self) -> list[str]:
        return self.editor.lines

    def draw(self, cv) -> None:
        h, w = cv.h, cv.w
        title = "BitScryer"
        what = (f"{self.work.number} {self.work.title}" if self.work.lesson else f"project: {self.work.title}")
        passed = "  ✓ passed" if self.work.lesson and self.progress.passed(self.work.key) else ""
        card = self.card.short() + f" ({self.sm})" if self.card else "no card"
        left = f" {title} · {what}{passed} · {card}"
        cv.put(0, 0, f" {title} ", COL["attn"])
        cv.put(0, len(title) + 2, left[len(title) + 2:][: w - len(title) - 2], COL["text"])
        st = ("⟳ " if self.busy else "") + ("✓ right" if self.status == "right" else self.status)
        colour = COL["ok"] if self.status == "right" else COL["bad"] if any(
            k in self.status for k in ("didn't", "failed", "not right")) else COL["warn"] if "not run" in \
            self.status else COL["muted"]
        room = w - len(left) - 4
        if room > 8:
            st = st if len(st) <= room else st[: room - 1] + "…"
            cv.put(0, w - len(st) - 1, st, colour)
        else:                                             # a narrow screen: the status where the keys are
            self.message = self.message or st
        res_h = 8
        top, bottom = 1, h - res_h - 1
        pane_h = max(3, bottom - top)
        src = self.source_lines()
        errs = {n for n, _ in (self.built.errors if self.built else []) if n}
        three = w >= 170 and self.work.source.suffix == ".cu"
        lw = int(w * 0.38) if three else max(40, min(w // 2, 100))
        self.draw_source(cv, top, 0, pane_h, lw, src, errs)
        if three:
            rw = (w - lw) // 2
            self.draw_asm(cv, top, lw, pane_h, rw, "ptx")
            self.draw_asm(cv, top, lw + rw, pane_h, w - lw - rw, "sass")
        else:
            self.draw_asm(cv, top, lw, pane_h, w - lw, self.right if self.work.source.suffix == ".cu" else "sass")
        self.draw_results(cv, bottom, 0, res_h, w)
        keys = ("‹^S›save+run ‹F1›lesson ‹F2›hint ‹F3›warps ‹F4›card ‹F5›run ‹F6›lessons ‹F7›output ‹F8›ptx/sass "
                "‹F9›keys ‹^Q›quit")
        keyed(cv, h - 1, 1, self.message or keys, COL["warn"] if self.message else COL["muted"], width=w - 2)
        if self.overlay:
            self.draw_overlay(cv)

    def draw_source(self, cv, y, x, h, w, src, errs) -> None:
        ed = self.editor
        state = "  ● not saved (Ctrl-S)" if ed.dirty else ""
        box(cv, y, x, h, w, f"{self.work.source.name}{state}")
        mapped = {ln.src for ln in self.built.sass if ln.src} if self.built else set()
        self.cursor_at = ed.draw(cv, y + 1, x + 1, h - 2, w - 2, COL, ptx=self.work.source.suffix == ".ptx",
                                 mapped=mapped, errs=errs, focused=not self.overlay)

    def draw_asm(self, cv, y, x, h, w, which) -> None:
        handwritten = self.work.source.suffix == ".ptx"
        lines = (self.built.ptx if which == "ptx" else self.built.sass) if self.built else []
        name = "PTX" if which == "ptx" else "SASS: the card's own instructions"
        n_ops = sum(1 for ln in lines if ln.kind == "op")
        box(cv, y, x, h, w, f"{name}  ({n_ops} instructions)" + ("" if handwritten else
                                                                  "  ‹Tab›" if w < 120 or which == "ptx" else ""))
        rows = h - 2
        if not lines:
            msg = "(builds when the file is saved)" if not self.built else "(nothing: it didn't build)"
            cv.put(y + 1, x + 2, msg, COL["muted"])
            return
        hits = [i for i, ln in enumerate(lines) if ln.src == self.cursor and not handwritten]
        start = max(0, hits[0] - 2) if hits else 0
        start = max(0, min(start, len(lines) - rows))
        for i in range(rows):
            j = start + i
            if j >= len(lines):
                break
            ln = lines[j]
            hit = j in hits
            fg = COL["attn"] if hit else COL["mem"] if ln.kind == "head" else COL["cyan"] if ln.kind == "label" \
                else COL["dim"] if ln.kind == "dir" else COL["text"]
            cv.put(y + 1 + i, x + 1, ("▶" if hit else " ") + ln.text[: w - 3], fg)
        if handwritten and which == "sass":
            cv.put(y + h - 1, x + 2, " hand-written PTX has no line marks, so nothing lights up ", COL["dim"])

    def draw_results(self, cv, y, x, h, w) -> None:
        box(cv, y, x, h, w, "results")
        rows: list[tuple[str, str]] = []
        b = self.built
        if b and not b.ok:
            for n, msg in b.errors[: h - 2]:
                rows.append(((f"line {n}: " if n else "") + msg, "bad"))
            if not b.errors:
                rows.append((b.log.strip().splitlines()[-1] if b.log.strip() else "it didn't build", "bad"))
        elif self.result and self.result.get("error"):
            err = self.result["error"]
            rows.append((err, "bad"))
            plain = explain(err)
            if plain:
                rows.append(("That usually means " + plain + ".", "warn"))
        elif self.result:
            for p in self.result.get("parts", []):
                rows.extend(self.part_lines(p, w - 4))
        elif b and b.ok:
            for k, u in b.usage.items():
                rows.append((f"{k}: {u['registers']} registers, {u['shared']} B shared, {u['stack']} B stack"
                             + (f", SPILLS {u['spill_stores']} B" if u["spill_stores"] else ""), "text"))
        out = []
        for text, c in rows:
            for piece in wrap(text, w - 4):
                out.append((piece, c))
        for i, (text, c) in enumerate(out[: h - 2]):
            cv.put(y + 1 + i, x + 2, text, COL.get(c, COL["text"]))

    def part_lines(self, p: dict, w: int) -> list[tuple[str, str]]:
        k = p.get("kernel", "?")
        st = p.get("status")
        if st == "missing":
            return [(f"{k}: {p.get('detail', 'not in the file yet')}", "muted")]
        if st == "error":
            return [(f"{k}: {p.get('detail')}", "bad")]
        out = []
        for name, c in (p.get("outputs") or {}).items():
            if c.get("right"):
                exact = "" if c.get("exact") is None or not c.get("n") else ", every one bit for bit" \
                    if c["exact"] == c["n"] else f", {c['exact'] / c['n']:.1%} bit for bit"
                out.append((f"{k}: ✓ right: {c.get('n', 0):,} values of {name}{exact}", "ok"))
            else:
                first = c.get("first") or []
                show = "; ".join(f"[{f['at']:,}] {f['got']:.6g} (want {f['want']:.6g})" for f in first[:3])
                out.append((f"{k}: ✗ {c.get('bad', 0):,} of {c.get('n', 0):,} values of {name} wrong. First: {show}"
                            if first else f"{k}: ✗ {c.get('detail', 'wrong')}", "bad"))
        if "ms_median" in p:
            bits = [f"{fmt_ms(p['ms_median'])} a run"]
            if p.get("gbs") is not None:
                peak = self.info.get("peak_bw", 0) / 1e9
                bits.append(f"{fmt_rate(p['gbs'], 'B/s')}" + (f" = {p['of_bw']:.0%} of {peak:.0f} GB/s"
                                                             if p.get("of_bw") else ""))
            if p.get("gflops") is not None:
                bits.append(f"{fmt_rate(p['gflops'], 'FLOPS')}" + (f" = {p['of_flops']:.0%} of the top"
                                                                   if p.get("of_flops") else ""))
            if p.get("goal"):
                got = p.get("of_bw") or p.get("of_flops") or 0
                bits.append(("★ goal met" if got >= p["goal"] else f"goal {p['goal']:.0%}"))
            out.append(("   " + " · ".join(bits), "cyan"))
            out.append((f"   {p.get('registers', 0)} registers · {p.get('shared', 0) + p.get('shared_dyn', 0)} B "
                        f"shared · {p.get('local', 0)} B local · {p.get('blocks_per_sm', 0)} blocks per SM · grid "
                        f"{'×'.join(map(str, p.get('grid', [])))} · block {'×'.join(map(str, p.get('block', [])))}",
                        "muted"))
        return out

    def overlay_lines(self, width: int) -> list[tuple[str, str]]:
        if self.overlay == "lesson":
            return markdown_lines(self.work.text(), width)
        if self.overlay == "hints":
            hints = self.work.hints()
            seen = self.progress.hints_seen(self.work.key)
            out: list[tuple[str, str]] = []
            for i, t in enumerate(hints[:seen]):
                out.append((f"Hint {i + 1} of {len(hints)}", "attn"))
                out.extend(markdown_lines(t, width))
                out.append(("", "text"))
            out.append(("‹h› another hint" if seen < len(hints) else "That's every hint.", "muted"))
            return out
        if self.overlay == "help":
            return markdown_lines(__doc__ or "", width)
        if self.overlay == "log":
            text = self.built.log if self.built else "(nothing built yet)"
            if self.result and self.result.get("trace"):
                text += "\n\n" + self.result["trace"]
            return [(ln, "text") for raw in text.splitlines() for ln in wrap(raw, width)]
        if self.overlay == "warps":
            parts = (self.result or {}).get("parts") or []
            parts = [p for p in parts if p.get("grid")]
            if not parts:
                return [("Run it on a card first: the picture is of the launch that ran.", "muted")]
            out = []
            for p in parts:
                out.extend(picture(p, self.info, width))
                out.append(("", "text"))
            return out
        return []

    def draw_overlay(self, cv) -> None:
        h, w = cv.h, cv.w
        oh, ow = h - 4, min(w - 4, 110)
        y, x = 2, (w - ow) // 2
        for i in range(oh):
            cv.put(y + i, x, " " * ow)
        titles = {"lesson": f"Lesson {self.work.number}: {self.work.title}", "hints": "Hints", "help": "Keys",
                  "log": "The build's output", "warps": "How the work splits up", "cards": "Cards",
                  "lessons": "Lessons"}
        box(cv, y, x, oh, ow, titles.get(self.overlay, ""))
        if self.overlay == "cards":
            return self.draw_cards(cv, y, x, oh, ow)
        if self.overlay == "lessons":
            return self.draw_lessons(cv, y, x, oh, ow)
        lines = []
        for t, c in self.overlay_lines(ow - 4):
            if len(t) <= ow - 4:
                lines.append((t, c))
            else:
                lead = " " * (len(t) - len(t.lstrip()))
                lines.extend((lead + piece if i else piece, c) for i, piece in
                             enumerate(textwrap.wrap(t, ow - 4 - len(lead), subsequent_indent="")))
        self.ov_scroll = max(0, min(self.ov_scroll, max(0, len(lines) - (oh - 3))))
        for i, (t, c) in enumerate(lines[self.ov_scroll: self.ov_scroll + oh - 3]):
            keyed(cv, y + 1 + i, x + 2, t, COL.get(c, COL["text"]), width=ow - 4)
        cv.put(y + oh - 2, x + 2, "‹↑↓› scroll  ‹Esc› back"[: ow - 4].replace("‹", "").replace("›", ""), COL["dim"])

    def draw_cards(self, cv, y, x, h, w) -> None:
        row = y + 1
        if not self.cards:
            cv.put(row, x + 2, "nvidia-smi shows no cards here.", COL["bad"])
        for i, c in enumerate(self.cards):
            why = self.allowed(c)
            on = i == self.ov_sel
            mine = self.card and c.uuid == self.card.uuid
            tc = "tensor cores" if c.tensor_cores else "no tensor cores"
            cv.put(row, x + 2, f"{'▶' if on else ' '} {'●' if mine else ' '} {c.short():<28} {c.sm}  {tc}",
                   COL["text"], COL["sel"] if on else None)
            cv.put(row + 1, x + 6, ("free to use" if not why else why)[: w - 8],
                   COL["ok"] if not why else COL["bad"] if cardlib.never(c, self.world) else COL["warn"])
            if self.info and mine:
                cv.put(row + 2, x + 6, f"top speed: {self.info.get('peak_bw', 0) / 1e9:.0f} GB/s memory, "
                       f"{self.info.get('peak_fp32', 0) / 1e12:.1f} TFLOPS fp32" +
                       (f", {self.info['peak_tensor'] / 1e12:.0f} TFLOPS tensor cores"
                        if self.info.get("peak_tensor") else ""), COL["muted"])
                row += 1
            row += 3
        row += 1
        n_ok = sum(1 for w_ in self.lessons if not ("tensor_cores" in w_.needs and not
                                                     (self.card and self.card.tensor_cores)))
        if self.card:
            cv.put(row, x + 2, f"On {self.card.short()}: {n_ok} of the {len(self.lessons)} lessons run.",
                   COL["text"])
        cv.put(y + h - 2, x + 2, "↑↓ choose  Enter use it  Esc back", COL["dim"])

    def draw_lessons(self, cv, y, x, h, w) -> None:
        for i, l in enumerate(self.lessons):
            on = i == self.ov_sel
            done = "✓" if self.progress.passed(l.key) else " "
            lock = "  (needs tensor cores: not on this card)" if "tensor_cores" in l.needs and not \
                (self.card and self.card.tensor_cores) else ""
            cur = "◀ open" if l.key == self.work.key else ""
            cv.put(y + 1 + i, x + 2, f"{'▶' if on else ' '} {done} {l.key[:2]}  {l.title}{lock}  {cur}"[: w - 4],
                   COL["muted"] if lock else COL["text"], COL["sel"] if on else None)
        cv.put(y + h - 2, x + 2, "↑↓ choose  Enter open  Esc back", COL["dim"])


def loop(stdscr, app: App) -> None:
    curses.raw()                                          # Ctrl-S, Ctrl-Q, Ctrl-Z and Ctrl-C come to the editor
    curses.use_default_colors()
    stdscr.timeout(150)
    stdscr.keypad(True)
    app.poll_file()
    app.start()
    last_check = time.time()
    while True:
        if time.time() - last_check > 0.4:
            last_check = time.time()
            if app.poll_file():
                app.start()
        stdscr.erase()
        cv = CursesCanvas(stdscr)
        try:
            app.draw(cv)
        except curses.error:
            pass
        try:
            if app.cursor_at and not app.overlay:
                curses.curs_set(1)
                stdscr.move(*app.cursor_at)
            else:
                curses.curs_set(0)
        except curses.error:
            pass
        stdscr.refresh()
        try:
            k = stdscr.get_wch()
        except curses.error:
            continue
        if k == curses.KEY_RESIZE:
            continue
        alt = False
        if k == "\x1b":                                   # Esc: on its own, or Alt-letter, or Esc then a letter
            stdscr.timeout(400)
            try:
                k2 = stdscr.get_wch()
            except curses.error:
                k2 = None
            stdscr.timeout(150)
            if isinstance(k2, str) and k2.isprintable():
                k, alt = k2.lower(), True
            elif k2 is not None:
                k = k2
        if not app.key(k, alt=alt):
            return


def selftest(card_pick: str | None) -> int:
    """Every lesson's answer (bitscryer/answers/) built and run on the card, in a folder of its own: does it work?"""
    app = App(work=None, progress=lessonlib.Progress(Path(tempfile.mkdtemp()) / "p.json"), card_pick=card_pick)
    print(f"Card: {app.card.short() + ' (' + app.sm + ')' if app.card else 'none free; building only'}")
    bad = ran = 0
    for w in app.lessons:
        ans = lessonlib.HERE / "answers" / f"{w.key}{w.source.suffix}"
        tmp = Path(tempfile.mkdtemp(prefix="bitscryer-"))
        work = lessonlib.Work(w.key, w.title, tmp, tmp / w.source.name, w.check, w.lesson, w.needs)
        shutil.copyfile(ans, work.source)
        app.work = work
        app.result = None
        app.cycle()
        line = f"{w.key:<22} {app.status}"
        for p in (app.result or {}).get("parts", []):
            bits = [p.get("kernel", "?"), p.get("status", "?")]
            if "ms_median" in p:
                bits.append(fmt_ms(p["ms_median"]))
            if p.get("of_bw"):
                bits.append(f"{p['of_bw']:.0%} of bandwidth")
            if p.get("of_flops"):
                bits.append(f"{p['of_flops']:.0%} of flops")
            line += "\n    " + " ".join(bits)
        if app.built and not app.built.ok:
            for n, msg in app.built.errors[:4] or [(0, (app.built.log.strip().splitlines() or ["?"])[-1])]:
                line += "\n    " + (f"line {n}: " if n else "") + msg[:300]
        if (app.result or {}).get("error"):
            line += "\n    " + app.result["error"]
        print(line, flush=True)
        if app.result and app.result.get("parts"):
            ran += 1
        if app.status != "right" and not (app.status.startswith("built") and "tensor" in app.status):
            bad += 1
        shutil.rmtree(tmp, ignore_errors=True)
    if not ran:
        print("Nothing ran on a card, so this only showed that everything builds.")
    print("All good." if not bad else f"{bad} didn't come out right: BitScryer (not you) needs a look.")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bitscryer", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("what", nargs="*", help="a project folder or file; or: new FOLDER")
    ap.add_argument("--card", help="a card's GPU number or UUID (it must be one BitScryer may use)")
    ap.add_argument("--once", action="store_true", help="draw one screen as text and stop")
    ap.add_argument("--size", default="140x45", help="--once's screen size, COLSxROWS")
    ap.add_argument("--selftest", action="store_true", help="every lesson's answer, built and run")
    ap.add_argument("--no-tmux", action="store_true")
    a = ap.parse_args(argv)
    if os.geteuid() == 0 and not os.environ.get("BITSCRYER_AS_ROOT"):
        print("bitscryer: run it as yourself, not with sudo (it needs nothing of root's)", file=sys.stderr)
        return 2
    for warning in load_colours():
        print(warning, file=sys.stderr)
    if a.selftest:
        return selftest(a.card)
    work = None
    if a.what[:1] == ["new"]:
        if len(a.what) < 2:
            ap.error("new FOLDER")
        work = lessonlib.new_project(Path(a.what[1]))
        print(f"Made {work.folder}: {work.source.name} and check.py. Opening it.")
        time.sleep(1)
    elif a.what:
        work = lessonlib.project(Path(a.what[0]))
    app = App(work, card_pick=a.card)
    if a.once:
        cols, rows = (int(v) for v in a.size.split("x"))
        app.poll_file()
        app.cycle()
        cv = TextCanvas(rows, cols)
        app.draw(cv)
        print(cv.text())
        return 0
    if not a.no_tmux and not os.environ.get("TMUX") and sys.stdin.isatty() and shutil.which("tmux"):
        os.execvp("tmux", ["tmux", "new-session", "-A", "-s", "bitscryer", "env", f"PYTHONPATH={CODE}",
                           sys.executable, "-m", "thaumaturge.bitscryer", *argv, "--no-tmux"])
    os.environ.setdefault("ESCDELAY", "25")
    curses.wrapper(loop, app)
    return 0


def console() -> None:
    """The bitscryer command, when installed with pip."""
    sys.exit(main(sys.argv[1:]))
