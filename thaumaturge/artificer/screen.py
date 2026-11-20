"""Artificer: build a model from snap-together blocks.

    artificer                     the model you had open last (at first: tiny-story, the small story model)
    artificer FOLDER              a model of your own: a folder with a .stack file and a blocks/ folder
    artificer new FOLDER          start a model, from tiny-story's stack and blocks
    artificer --check FOLDER      check a model and print what it found

On the right, the stack: the model's blocks, top to bottom, with the shape each gives and its weights. On the left,
the code of the block you open (Enter), in the same editor as BitScryer's. Ctrl-S saves it and checks the model.

A check builds the model on the CPU and sends a few words through it (thaumaturge/artificer/check.py), in a program
of its own with a time limit, so a block's mistake never takes the screen down. Nothing here uses a card yet.
"""

from __future__ import annotations

import ast
import curses
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..common.editor import Editor
from ..common.screen import COL, CursesCanvas, TextCanvas, box, config_dir, into_tmux, keyed, load_colours, width
from . import stack as S
from .check import LIBRARY, count, key, size, where

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STARTERS = HERE / "starter"
CHECK_LIMIT = 120                      # seconds a check may take
NAME_OK = re.compile(r"^[a-z][a-z0-9-]{0,39}$")
STATE = "artificer.json"               # in ~/.config/thaumaturge: the model open last


def home() -> Path:
    return Path(os.environ.get("ARTIFICER_HOME") or Path.home() / "artificer")


# -- models (folders) and the blocks they can use

def new_model(folder: Path, starter: str = "tiny-story") -> Path:
    """A new model folder: the starter's stack, and every block it uses copied into blocks/ to change freely."""
    folder = Path(folder).expanduser()
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f"{folder} isn't empty: pick a new folder")
    src = STARTERS / f"{starter}.stack"
    (folder / "blocks").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, folder / f"{folder.name}.stack")
    for nd in S.walk(S.parse(src.read_text()).nodes):
        if nd.kind == "block" and (LIBRARY / f"{nd.name}.py").exists():
            shutil.copyfile(LIBRARY / f"{nd.name}.py", folder / "blocks" / f"{nd.name}.py")
    return folder


def describe(path: Path) -> dict:
    """A block file's plugs and first line of description, read without running it."""
    info = {"name": path.stem, "takes": "?", "gives": "?", "about": "", "path": path}
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        info["about"] = "(doesn't read as Python yet)"
        return info
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        doc = ast.get_docstring(cls) or ""
        info["about"] = doc.strip().split("\n")[0]
        for st in cls.body:
            if isinstance(st, ast.Assign):
                names = [t for t in st.targets]
                for t in names:
                    if isinstance(t, ast.Tuple) and isinstance(st.value, ast.Tuple):
                        for a, b in zip(t.elts, st.value.elts):
                            if isinstance(a, ast.Name) and isinstance(b, ast.Constant):
                                info[a.id] = b.value
                    elif isinstance(t, ast.Name) and isinstance(st.value, ast.Constant):
                        info[t.id] = st.value.value
        break
    return info


def available(blocks: Path) -> list[dict]:
    """The blocks a model can add: its own (blocks/), then the library's it doesn't have yet."""
    out, have = [], set()
    for p in sorted(blocks.glob("*.py")):
        d = describe(p)
        d["where"] = "yours"
        out.append(d)
        have.add(p.stem)
    for p in sorted(LIBRARY.glob("*.py")):
        if p.stem in have or p.stem == "template":
            continue
        d = describe(p)
        d["where"] = "library"
        out.append(d)
    return out


def class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("-")) or "Block"


def run_check(folder: Path) -> dict:
    """check.py on the model, in a program of its own: the CPU only, nice 10, a time limit."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        p = subprocess.run([sys.executable, "-m", "thaumaturge.artificer.check", str(folder), "--json"],
                           capture_output=True, text=True, timeout=CHECK_LIMIT, env=env,
                           preexec_fn=lambda: os.nice(10))
    except subprocess.TimeoutExpired:
        return {"ok": False, "nodes": {}, "totals": {}, "problems": [
            {"path": "", "msg": f"the check took over {CHECK_LIMIT} s and was stopped: is a block stuck in a loop?",
             "file": None, "line": 0}]}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        err = (p.stderr.strip().splitlines() or ["no answer"])[-1]
        if "No module named 'torch'" in p.stderr:
            err = "PyTorch isn't installed for this Python (install.sh puts it in)"
        return {"ok": False, "nodes": {}, "totals": {}, "problems": [
            {"path": "", "msg": f"the check didn't finish: {err}", "file": None, "line": 0}]}


# -- the screen

HELP = """Artificer builds a model from blocks, top to bottom: words go in at the top as numbers (tokens), each
block passes rows of numbers (width) to the next, and the last gives a score for every word (logits).

Two blocks fit when what the first gives is what the next takes. "add" is a shortcut: what goes in comes out
again, with what the blocks inside it make of it added on, so each block only has to learn a change. "repeat 12"
stacks the blocks inside it 12 times, each copy with weights of its own.

The stack (on the right)
  ↑ ↓          choose                    Enter      open the block's code on the left
  a            add a block after this    n          a new block of your own, from a template
  Shift-↑ ↓    move it (also [ and ])    c          copy it         d   remove it
  g            put it inside a shortcut (add)       r   repeat it (on a repeat: how many times)
  x            take a group away, keeping what's inside              =   this block's settings (heads=8)
  s            check the model now       z          undo            e   the stack file itself, as text
  ?            this                      q          quit

The code (on the left)
  Ctrl-S saves and checks. Esc goes back to the stack. Ctrl-Z/Y undo/redo, Ctrl-F find, Ctrl-G go to a line,
  Ctrl-C/X/V copy, cut, paste, Ctrl-K delete the line, Ctrl-D duplicate it.

F1 this, F2 check, F5 the stack file, F6 code or stack, F10 or Ctrl-Q quit. Where F-keys don't come through,
Esc then the letter does the same (Esc then s checks).

A check runs on the CPU only: nothing in Artificer uses a card yet. The numbers it gives are for fp16 on a card;
"to train" is rough (weights, Adam and the rest at about 16 bytes a weight, plus what each block keeps)."""


class App:
    COMMANDS = {curses.KEY_F1: "?", curses.KEY_F2: "s", curses.KEY_F5: "e", curses.KEY_F6: "\t",
                curses.KEY_F10: "q"}

    def __init__(self, folder: Path, check_now: bool = True):
        self.folder = Path(folder).expanduser().resolve()
        self.stack_path, self.blocks = where(self.folder)
        self.st = S.parse(self.read_stack())
        self.sel = 0
        self.focus = "stack"
        self.editor = Editor(text="")
        self.editing: Path | None = None
        self.result: dict | None = None
        self.checking = False
        self.again = False
        self.lock = threading.Lock()
        self.undo: list[str] = []
        self.overlay: str | None = None
        self.ov_sel = 0
        self.ov_scroll = 0
        self.choices: list[dict] = []
        self.prompt: tuple[str, str] | None = None     # (what's asked, the answer so far)
        self.message = ""
        self.cursor_at = None
        rows = self.rows()
        if rows:
            first = next((i for i, (_, _, nd) in enumerate(rows) if nd.kind == "block"), 0)
            self.sel = first
            self.open_selected(focus=False)
        else:
            self.open_file(self.stack_path, focus=False)
        if check_now:
            self.start_check()

    # -- the stack

    def read_stack(self) -> str:
        try:
            return self.stack_path.read_text()
        except OSError:
            return ""

    def rows(self):
        return S.rows(self.st)

    def selected(self):
        rows = self.rows()
        if not rows:
            return None, None
        self.sel = max(0, min(self.sel, len(rows) - 1))
        p, _, nd = rows[self.sel]
        return p, nd

    def select_path(self, path) -> None:
        for i, (p, _, _) in enumerate(self.rows()):
            if p == path:
                self.sel = i
                return

    def change(self, fn, *args) -> None:
        """A change to the stack through the screen: kept for undo, written, checked."""
        if self.st.errors:
            self.message = "The stack file has a mistake in it: fix it as text first (e)."
            return
        before = S.write(self.st)
        path = fn(*args)
        after = S.write(self.st)
        if after == before:
            return
        self.undo.append(before)
        del self.undo[:-200]
        self.write_stack(after)
        if path is not None:
            self.select_path(tuple(path))

    def write_stack(self, text: str) -> None:
        tmp = self.stack_path.with_name(self.stack_path.name + ".saving")
        tmp.write_text(text)
        tmp.replace(self.stack_path)
        self.st = S.parse(text)
        if self.editing == self.stack_path and not self.editor.dirty:
            self.editor.set_text(text)
        self.start_check()

    # -- the editor

    def open_file(self, path: Path, focus: bool = True) -> None:
        if self.editor.dirty and self.editing:
            self.editor.save()
            self.after_save()
        self.editing = path
        self.editor = Editor(path)
        if focus:
            self.focus = "code"

    def open_selected(self, focus: bool = True) -> None:
        p, nd = self.selected()
        if nd is None:
            return
        if nd.kind != "block":
            self.message = f"{nd.label()} is a group: it has no code of its own. Open a block inside it."
            return
        f = self.blocks / f"{nd.name}.py"
        if not f.exists():
            self.message = f"There's no blocks/{f.name} yet: add it with a, or make one with n."
            return
        self.open_file(f, focus)

    def after_save(self) -> None:
        if self.editing == self.stack_path:
            self.st = S.parse(self.editor.text())
        self.start_check()

    # -- checking

    def start_check(self) -> None:
        with self.lock:
            if self.checking:
                self.again = True
                return
            self.checking = True
        threading.Thread(target=self._check, daemon=True).start()

    def _check(self) -> None:
        while True:
            res = run_check(self.folder)
            with self.lock:
                self.result = res
                if not self.again:
                    self.checking = False
                    return
                self.again = False

    def wait(self) -> None:
        while self.checking:
            time.sleep(0.05)

    # -- keys

    def key(self, k, alt: bool = False) -> bool:
        """One key, as get_wch gives it (alt: it came after Esc). False to quit."""
        self.message = ""
        if k == "\x11":                                  # Ctrl-Q, anywhere
            return self.quit()
        if self.prompt is not None:
            return self.key_prompt(k)
        if self.overlay:
            return self.key_overlay(k)
        cmd = self.COMMANDS.get(k)
        if cmd == "\t":
            self.focus = "stack" if self.focus == "code" else "code"
            return True
        if cmd:
            return self.command(cmd)
        if self.focus == "code":
            if alt and isinstance(k, str):
                self.focus = "stack"
                return self.command(k)
            if k == "\x1b":
                self.focus = "stack"
                return True
            if self.editor.key(k):
                self.message = self.editor.message
                if k == "\x13" and not self.editor.dirty:
                    self.after_save()
                return True
            return True
        return self.command(k)

    def quit(self) -> bool:
        if self.editor.dirty:
            self.editor.save()
        return False

    def command(self, k) -> bool:
        rows = self.rows()
        p, nd = self.selected()
        if k in ("q",):
            return self.quit()
        if k in (curses.KEY_DOWN, "j"):
            self.sel = min(len(rows) - 1, self.sel + 1)
        elif k in (curses.KEY_UP, "k"):
            self.sel = max(0, self.sel - 1)
        elif k == curses.KEY_NPAGE:
            self.sel = min(len(rows) - 1, self.sel + 10)
        elif k == curses.KEY_PPAGE:
            self.sel = max(0, self.sel - 10)
        elif k == curses.KEY_HOME:
            self.sel = 0
        elif k == curses.KEY_END:
            self.sel = max(0, len(rows) - 1)
        elif k in ("\n", "\r", curses.KEY_ENTER, 10, 13):
            self.open_selected()
        elif k in (curses.KEY_SR, "[") and p:
            self.change(S.move, self.st, p, -1)
        elif k in (curses.KEY_SF, "]") and p:
            self.change(S.move, self.st, p, +1)
        elif k == "a":
            self.choices = available(self.blocks)
            self.overlay, self.ov_sel = "add", 0
        elif k == "n":
            self.prompt = ("new block's name (like my-attention)", "")
        elif k == "d" and p:
            self.prompt = (f"remove {nd.label()}{' and everything inside it' if nd.group else ''}? y or n", "")
        elif k == "c" and p:
            self.change(S.duplicate, self.st, p)
        elif k == "g" and p:
            self.change(S.wrap, self.st, p, "add")
        elif k == "r" and p:
            if nd.kind == "repeat":
                self.prompt = ("repeat how many times", str(nd.count))
            else:
                self.prompt = ("repeat it how many times", "2")
        elif k == "x" and p:
            if nd.group:
                self.change(S.unwrap, self.st, p)
            else:
                self.message = "x takes a group (add or repeat) away; this is a block."
        elif k == "=" and p:
            if nd.kind == "block":
                self.prompt = (f"{nd.name}'s own settings (name=value, name= to drop one)",
                               " ".join(f"{a}={S.show(b)}" for a, b in nd.settings.items()))
            elif nd.kind == "repeat":
                self.prompt = ("repeat how many times", str(nd.count))
        elif k == "z":
            if not self.undo:
                self.message = "Nothing to undo."
            else:
                self.write_stack(self.undo.pop())
                self.message = "Undone."
        elif k == "s":
            self.start_check()
        elif k == "e":
            self.open_file(self.stack_path)
        elif k == "?":
            self.overlay, self.ov_scroll = "help", 0
        return True

    def key_prompt(self, k) -> bool:
        ask, ans = self.prompt
        if k == "\x1b":
            self.prompt = None
            return True
        if ask.endswith("y or n"):
            self.prompt = None
            if k in ("y", "Y"):
                p, _ = self.selected()
                self.change(S.remove, self.st, p)
            return True
        if k in (curses.KEY_BACKSPACE, "\x7f", "\x08", 127, 8):
            self.prompt = (ask, ans[:-1])
            return True
        if isinstance(k, str) and len(k) == 1 and k.isprintable():
            self.prompt = (ask, ans + k)
            return True
        if k not in ("\n", "\r", curses.KEY_ENTER, 10, 13):
            return True
        self.prompt = None
        p, nd = self.selected()
        if ask.startswith("new block"):
            self.new_block(ans.strip())
        elif ask.startswith("repeat it"):
            try:
                n = int(ans)
                assert n >= 1
            except (ValueError, AssertionError):
                self.message = "A whole number, 1 or more."
                return True
            self.change(S.wrap, self.st, p, "repeat", n)
        else:
            bad = []

            def set_it(path):
                bad.extend(S.set_settings(S.node(self.st, path), ans))
                return path
            self.change(set_it, p)
            if bad:
                self.message = "; ".join(bad)
        return True

    def new_block(self, name: str) -> None:
        if not NAME_OK.match(name):
            self.message = "A name in small letters, numbers and dashes, starting with a letter (like my-norm)."
            return
        f = self.blocks / f"{name}.py"
        if f.exists():
            self.message = f"blocks/{f.name} is there already: add it with a."
            return
        self.blocks.mkdir(parents=True, exist_ok=True)
        f.write_text((LIBRARY / "template.py").read_text().replace("class NAME(", f"class {class_name(name)}("))
        p, _ = self.selected()
        self.change(S.insert, self.st, p, S.Node("block", name=name))
        self.open_file(f)
        self.message = f"blocks/{f.name}: change it, Ctrl-S to check."

    def key_overlay(self, k) -> bool:
        if k in ("\x1b", "q") or (self.overlay == "help" and k in ("?", curses.KEY_F1)):
            self.overlay = None
            return True
        if self.overlay == "help":
            if k in (curses.KEY_DOWN, "j"):
                self.ov_scroll += 1
            elif k in (curses.KEY_UP, "k"):
                self.ov_scroll = max(0, self.ov_scroll - 1)
            return True
        n = len(self.choices)
        if k in (curses.KEY_DOWN, "j"):
            self.ov_sel = min(n - 1, self.ov_sel + 1)
        elif k in (curses.KEY_UP, "k"):
            self.ov_sel = max(0, self.ov_sel - 1)
        elif k in ("\n", "\r", curses.KEY_ENTER, 10, 13) and n:
            c = self.choices[self.ov_sel]
            self.overlay = None
            if c["where"] == "library":
                self.blocks.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(c["path"], self.blocks / f"{c['name']}.py")
            p, _ = self.selected()
            self.change(S.insert, self.st, p, S.Node("block", name=c["name"]))
            self.message = f"Added {c['name']}" + (" (copied from the library into blocks/)" if c["where"] ==
                                                   "library" else "")
        return True

    # -- drawing

    def problems(self) -> list[dict]:
        return (self.result or {}).get("problems", [])

    def draw(self, cv) -> None:
        h, w = cv.h, cv.w
        res = self.result
        title = " Artificer "
        cv.put(0, 0, title, COL["attn"])
        cv.put(0, len(title), f"· {self.folder.name}", COL["text"])
        if self.checking:
            st, colour = "⟳ checking", COL["muted"]
        elif res is None:
            st, colour = "not checked", COL["muted"]
        elif res.get("ok"):
            st, colour = f"✓ fits · {count(res['totals']['weights'])} weights", COL["ok"]
        else:
            n = len(res.get("problems", []))
            st, colour = f"✗ {n} problem{'s' if n != 1 else ''}", COL["bad"]
        cv.put(0, max(len(title) + 2, w - len(st) - 1), st, colour)
        top, bottom = 1, h - 1
        lw = max(40, min(w // 2, 100)) if w >= 90 else w
        if w >= 90:
            self.draw_code(cv, top, 0, bottom - top, lw)
            self.draw_stack(cv, top, lw, bottom - top, w - lw)
        elif self.focus == "code":
            self.draw_code(cv, top, 0, bottom - top, w)
        else:
            self.draw_stack(cv, top, 0, bottom - top, w)
        if self.prompt:
            ask, ans = self.prompt
            line = f"{ask}: {ans}"
            cv.put(h - 1, 0, (" " + line + "   (Enter, Esc)")[:w].ljust(w), COL["key"])
            self.cursor_at = (h - 1, min(w - 1, len(line) + 1))
        else:
            if self.focus == "code":
                keys = "‹^S› save+check  ‹Esc› back to the stack  ‹F2› check  ‹F5› stack file  ‹F1› help  ‹^Q› quit"
            else:
                keys = ("‹Enter› code  ‹a›dd ‹n›ew ‹d›elete ‹c›opy  ‹[ ]› move  ‹g› shortcut  ‹r›epeat  ‹x› ungroup  "
                        "‹=› settings  ‹s› check  ‹z› undo  ‹?› help  ‹q›uit")
            keyed(cv, h - 1, 1, self.message or keys, COL["warn"] if self.message else COL["muted"], width=w - 2)
        if self.overlay == "add":
            self.draw_add(cv)
        elif self.overlay == "help":
            self.draw_help(cv)

    def draw_code(self, cv, y, x, h, w) -> None:
        ed = self.editor
        name = self.editing.name if self.editing else "(nothing open)"
        if self.editing and self.editing.parent == self.blocks:
            name = f"blocks/{name}"
        state = "  ● not saved (Ctrl-S)" if ed.dirty else ""
        box(cv, y, x, h, w, name + state)
        errs = set()
        if self.editing:
            for p in self.problems():
                if p.get("file") == self.editing.name and p.get("line"):
                    errs.add(p["line"])
        at = ed.draw(cv, y + 1, x + 1, h - 2, w - 2, COL, errs=errs,
                     focused=self.focus == "code" and not self.overlay,
                     python=bool(self.editing and self.editing.suffix == ".py"))
        if self.focus == "code" and not self.overlay and not self.prompt:
            self.cursor_at = at
        elif not self.prompt:
            self.cursor_at = None

    def node_line(self, path, depth, nd, w) -> tuple[str, str, str]:
        """A stack row: (left part, right part, colour)."""
        res = self.result or {}
        info = res.get("nodes", {}).get(key(path), {})
        guide = "│ " * depth
        label = nd.label()
        if nd.kind == "block" and nd.settings:
            label += " " + " ".join(f"{a}={S.show(b)}" for a, b in nd.settings.items())
        bits = []
        if nd.kind == "block":
            shape = info.get("shape")
            plug = f"{info.get('takes', '?')} → {info.get('gives', '?')}" if "takes" in info else ""
            if shape and info.get("gives") == "width":
                plug = f"→ {shape[-1]} wide"
            elif shape and info.get("gives") == "logits":
                plug = f"→ {shape[-1]} words"
            elif shape and info.get("gives") == "tokens":
                plug = "→ tokens"
            bits.append(plug)
            if info.get("shares"):
                bits.append(f"uses {info['shares']}'s")
            elif "weights" in info:
                copies = info.get("copies", 1)
                bits.append(count(info["weights"]) + (f" ({copies}×{count(info['each'])})" if copies > 1 else ""))
        elif "weights" in info:
            bits.append(count(info["weights"]))
        return guide + label, "  ".join(b for b in bits if b), COL["text"]

    def draw_stack(self, cv, y, x, h, w) -> None:
        box(cv, y, x, h, w, f"the stack · {self.stack_path.name}")
        rows = self.rows()
        res = self.result or {}
        bad_paths = {p["path"] for p in res.get("problems", []) if p.get("path")}
        foot = self.foot_lines(w - 4)
        list_h = h - 2 - len(foot) - (1 if foot else 0)
        if self.st.errors:
            for i, (ln, msg) in enumerate(self.st.errors[:list_h]):
                cv.put(y + 1 + i, x + 2, f"✗ line {ln}: {msg}"[: w - 4], COL["bad"])
        else:
            first = max(0, min(self.sel - list_h // 2, len(rows) - list_h))
            for i, (p, depth, nd) in enumerate(rows[first:first + list_h]):
                n = first + i
                left, right, colour = self.node_line(p, depth, nd, w)
                k = key(p)
                if k in bad_paths:
                    colour = COL["bad"]
                    left = left + " ✗"
                elif nd.group:
                    colour = COL["mem"] if nd.kind == "repeat" else COL["cyan"]
                sel = n == self.sel
                bg = COL["sel"] if sel and self.focus == "stack" and not self.overlay else None
                inner = w - 4
                text = left[: inner - width(right) - 1].ljust(inner - width(right)) + right
                cv.put(y + 1 + i, x + 1, ("▶" if sel else " ") + text[:inner] + " ", colour, bg)
                if right and k not in bad_paths:
                    cv.put(y + 1 + i, x + 2 + inner - width(right), right, COL["muted"], bg)
                guide = "│ " * depth
                if depth:
                    cv.put(y + 1 + i, x + 2, guide, COL["dim"], bg)
        if foot:
            fy = y + h - 1 - len(foot)
            cv.put(fy - 1, x + 1, "─" * (w - 2), COL["frame"])
            for i, (text, colour) in enumerate(foot):
                cv.put(fy + i, x + 2, text[: w - 4], colour)

    def foot_lines(self, w: int) -> list[tuple[str, str]]:
        res = self.result
        if res is None:
            return []
        out = []
        if res.get("ok"):
            t = res["totals"]
            out.append((f"{count(t['weights'])} weights · to run, in fp16: {size(t['run_bytes'])}", COL["ok"]))
            out.append((f"to train, batch {t['train_batch']} × {t['context']} words: about "
                        f"{size(t['train_bytes'])} (rough)", COL["muted"]))
        for p in res.get("problems", [])[:6]:
            where_ = f"{p['file']}:{p['line']} " if p.get("file") and p.get("line") else ""
            msg = f"✗ {where_}{p['msg']}"
            while msg and len(out) < 8:
                out.append((msg[:w], COL["bad"]))
                msg = ("  " + msg[w:]) if len(msg) > w else ""
        return out

    def draw_add(self, cv) -> None:
        h, w = cv.h, cv.w
        bw = min(w - 4, 96)
        bh = min(h - 4, len(self.choices) + 4)
        y, x = (h - bh) // 2, (w - bw) // 2
        for i in range(bh):
            cv.put(y + i, x, " " * bw)
        box(cv, y, x, bh, bw, "add a block after the one chosen  ‹Enter› add  ‹Esc› back")
        first = max(0, self.ov_sel - (bh - 4) + 1)
        for i, c in enumerate(self.choices[first:first + bh - 4]):
            n = first + i
            plug = f"{c['takes']}→{c['gives']}"
            line = f" {c['name']:<16} {plug:<16} {c['about']}"
            tag = "" if c["where"] == "yours" else " (library)"
            line = line[: bw - 4 - len(tag)].ljust(bw - 4 - len(tag)) + tag
            cv.put(y + 2 + i, x + 1, line + " ", COL["text"], COL["sel"] if n == self.ov_sel else None)
        if not self.choices:
            cv.put(y + 2, x + 2, "No blocks found: n makes one.", COL["muted"])

    def draw_help(self, cv) -> None:
        h, w = cv.h, cv.w
        bw = min(w - 2, 118)
        lines = []
        for para in HELP.split("\n"):
            while len(para) > bw - 4:
                cut = para.rfind(" ", 0, bw - 4)
                cut = cut if cut > 0 else bw - 4
                lines.append(para[:cut])
                para = para[cut:].lstrip()
            lines.append(para)
        bh = min(h - 2, len(lines) + 2)
        y, x = (h - bh) // 2, (w - bw) // 2
        for i in range(bh):
            cv.put(y + i, x, " " * bw)
        box(cv, y, x, bh, bw, "Artificer  ‹Esc› back")
        self.ov_scroll = max(0, min(self.ov_scroll, len(lines) - (bh - 2)))
        for i, line in enumerate(lines[self.ov_scroll:self.ov_scroll + bh - 2]):
            cv.put(y + 1 + i, x + 2, line, COL["text"])


def loop(stdscr, app: App) -> None:
    curses.raw()                                          # Ctrl-S, Ctrl-Q, Ctrl-Z and Ctrl-C come to the editor
    curses.use_default_colors()
    stdscr.timeout(150)
    stdscr.keypad(True)
    while True:
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
        if k == "\x1b":                                   # Esc: on its own, or Esc then a letter
            stdscr.timeout(400)
            try:
                k2 = stdscr.get_wch()
            except curses.error:
                k2 = None
            stdscr.timeout(150)
            if isinstance(k2, str) and k2.isprintable():
                k, alt = k2, True
            elif k2 is not None:
                k = k2
        if not app.key(k, alt=alt):
            return


def remember(folder: Path) -> None:
    try:
        d = config_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / STATE).write_text(json.dumps({"model": str(folder)}))
    except OSError:
        pass


def last_model() -> Path | None:
    try:
        p = Path(json.loads((config_dir() / STATE).read_text())["model"])
        return p if p.exists() else None
    except (OSError, ValueError, KeyError):
        return None


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="artificer", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("what", nargs="*", help="a model's folder; or: new FOLDER")
    ap.add_argument("--check", action="store_true", help="check the model and print what it found")
    ap.add_argument("--once", action="store_true", help="draw one screen as text and stop")
    ap.add_argument("--size", default="140x45", help="--once's screen size, COLSxROWS")
    ap.add_argument("--no-tmux", action="store_true")
    a = ap.parse_args(argv)
    if os.geteuid() == 0 and not os.environ.get("ARTIFICER_AS_ROOT"):
        print("artificer: run it as yourself, not with sudo (it needs nothing of root's)", file=sys.stderr)
        return 2
    for warning in load_colours():
        print(warning, file=sys.stderr)
    if a.what[:1] == ["new"]:
        if len(a.what) < 2:
            ap.error("new FOLDER")
        folder = new_model(Path(a.what[1]))
        print(f"Made {folder}: {folder.name}.stack, and its blocks in blocks/.")
        if not (a.check or a.once):
            time.sleep(1)
    elif a.what:
        folder = Path(a.what[0]).expanduser()
        if not folder.is_dir():
            ap.error(f"{folder} isn't a folder (artificer new {folder} starts one)")
    else:
        folder = last_model() or home() / "tiny-story"
        if not folder.exists():
            new_model(folder)
            print(f"Made {folder} to start from: the small story model.")
    folder = folder.resolve()
    if a.check:
        from .check import report
        res = run_check(folder)
        print(report(res))
        return 0 if res["ok"] else 1
    if a.once:
        cols, rows = (int(v) for v in a.size.lower().split("x"))
        app = App(folder)
        app.wait()
        cv = TextCanvas(rows, cols)
        app.draw(cv)
        print(cv.text(colour=sys.stdout.isatty()))
        return 0
    remember(folder)
    if not a.no_tmux:
        into_tmux("artificer", [sys.executable, "-m", "thaumaturge.artificer", "--no-tmux", str(folder)])
    app = App(folder)
    curses.wrapper(loop, app)
    return 0


def console() -> None:
    """The artificer command, when installed with pip."""
    sys.exit(main(sys.argv[1:]))
