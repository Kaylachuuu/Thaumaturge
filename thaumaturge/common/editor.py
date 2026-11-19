"""BitScryer's own editor: your file, edited right in the left pane.

Keys (as in most editors):

    arrows, Home, End, PgUp, PgDn          move          Shift + arrows, Home, End    select
    Ctrl-S  save (and so build and run)    Ctrl-Z  undo                   Ctrl-Y  redo
    Ctrl-C  copy (the line, if nothing is selected)        Ctrl-X  cut (likewise)    Ctrl-V  paste
    Ctrl-K  delete the line               Ctrl-D  duplicate the line     Ctrl-F  find (again: the next one)
    Ctrl-G  go to a line                  Tab / Shift-Tab  indent / take it back      Enter keeps the indent

No curses in here: the keys come in as get_wch gives them (a str for a character, an int for a special key), so it
can be tested without a terminal. Drawing goes through a canvas (thaumaturge/common/screen.py).
"""

from __future__ import annotations

import curses
import re
from pathlib import Path

INDENT = "    "
UNDO_KEEP = 500

# Shift with arrows and the like, as curses names them (and as xterm sends them, where curses has no name).
S_LEFT, S_RIGHT, S_UP, S_DOWN = curses.KEY_SLEFT, curses.KEY_SRIGHT, curses.KEY_SR, curses.KEY_SF
S_HOME, S_END = curses.KEY_SHOME, curses.KEY_SEND
BTAB = curses.KEY_BTAB

CUDA_WORDS = set("""__global__ __device__ __host__ __shared__ __constant__ __restrict__ __syncthreads __launch_bounds__
extern static const inline void int unsigned signed long short char float double bool half half2 float2 float4 int4
uint2 uint4 size_t if else for while do return break continue switch case default struct template typename
using namespace constexpr auto sizeof true false nullptr define include pragma unroll""".split())
BUILTINS = set("threadIdx blockIdx blockDim gridDim warpSize".split())
TOKEN = re.compile(r"(//.*$)|(\"(?:[^\"\\\\]|\\\\.)*\")|(\b0[xX][0-9a-fA-F]+\b|\b\d+\.?\d*[fFuUlL]*\b)|"
                   r"(%[A-Za-z_][\w.]*|[#.]?[A-Za-z_]\w*)")


def colour_spans(line: str, ptx: bool) -> list[tuple[int, int, str]]:
    """(start, end, colour name) for the parts of a line worth colouring: comments, strings, numbers, keywords,
    the built-in thread numbers, and in PTX its directives (.reg), registers (%r1) and opcodes."""
    out = []
    for m in TOKEN.finditer(line):
        s, e = m.span()
        if m.group(1):
            out.append((s, e, "muted"))
            break
        if m.group(2):
            out.append((s, e, "ff"))
        elif m.group(3):
            out.append((s, e, "cyan"))
        else:
            word = m.group(4)
            if ptx:
                if word.startswith("%"):
                    out.append((s, e, "mem"))
                elif word.startswith("."):
                    out.append((s, e, "attn"))
            elif word.lstrip("#") in CUDA_WORDS:
                out.append((s, e, "attn"))
            elif word in BUILTINS:
                out.append((s, e, "mem"))
    return out


class Editor:
    def __init__(self, path: Path | None = None, text: str = ""):
        self.path = path
        self.lines = [""]
        self.row = self.col = 0          # where the cursor is (0-based)
        self.want = 0                    # the column up and down try to keep
        self.top = self.left = 0         # the first line and column on screen
        self.anchor: tuple[int, int] | None = None   # the other end of the selection
        self.undo: list[tuple] = []
        self.redo: list[tuple] = []
        self.clip = ""
        self.clip_lines = False          # the clipboard holds whole lines (copied with nothing selected)
        self.dirty = False
        self.prompt: str | None = None   # "find" or "line" while asking
        self.answer = ""
        self.last_find = ""
        self.message = ""
        self.height = 20
        self.last_kind = ""              # typing runs into one undo step
        if path is not None:
            self.load()
        else:
            self.set_text(text)

    # -- the file

    def set_text(self, text: str) -> None:
        self.lines = text.replace("\r\n", "\n").replace("\t", INDENT).split("\n")
        if len(self.lines) > 1 and self.lines[-1] == "":
            self.lines.pop()
        self.lines = self.lines or [""]
        self.row = min(self.row, len(self.lines) - 1)
        self.col = min(self.col, len(self.lines[self.row]))
        self.dirty = False

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def load(self) -> None:
        try:
            self.set_text(self.path.read_text(errors="replace"))
        except OSError:
            self.set_text("")

    def save(self) -> bool:
        try:
            tmp = self.path.with_name(self.path.name + ".saving")
            tmp.write_text(self.text())
            tmp.replace(self.path)
        except OSError as e:
            self.message = f"couldn't save: {e}"
            return False
        self.dirty = False
        self.message = "saved"
        return True

    # -- undo

    def snapshot(self, kind: str) -> None:
        """Before a change: what to go back to. Letters typed one after another make one step."""
        if kind == "type" and self.last_kind == "type":
            return
        self.undo.append((list(self.lines), self.row, self.col))
        del self.undo[:-UNDO_KEEP]
        self.redo.clear()
        self.last_kind = kind

    def _swap(self, src: list, dst: list) -> None:
        if not src:
            self.message = "nothing to undo" if src is self.undo else "nothing to redo"
            return
        dst.append((list(self.lines), self.row, self.col))
        self.lines, self.row, self.col = src.pop()
        self.anchor = None
        self.dirty = True
        self.last_kind = ""

    # -- selection

    def selection(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if self.anchor is None or self.anchor == (self.row, self.col):
            return None
        a, b = self.anchor, (self.row, self.col)
        return (a, b) if a <= b else (b, a)

    def selected_text(self) -> str:
        sel = self.selection()
        if not sel:
            return ""
        (r0, c0), (r1, c1) = sel
        if r0 == r1:
            return self.lines[r0][c0:c1]
        return "\n".join([self.lines[r0][c0:], *self.lines[r0 + 1:r1], self.lines[r1][:c1]])

    def delete_selection(self) -> bool:
        sel = self.selection()
        if not sel:
            self.anchor = None
            return False
        (r0, c0), (r1, c1) = sel
        self.lines[r0:r1 + 1] = [self.lines[r0][:c0] + self.lines[r1][c1:]]
        self.row, self.col = r0, c0
        self.anchor = None
        self.dirty = True
        return True

    def insert(self, s: str) -> None:
        s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", INDENT)
        line = self.lines[self.row]
        before, after = line[:self.col], line[self.col:]
        parts = s.split("\n")
        if len(parts) == 1:
            self.lines[self.row] = before + s + after
            self.col += len(s)
        else:
            new = [before + parts[0], *parts[1:-1], parts[-1] + after]
            self.lines[self.row:self.row + 1] = new
            self.row += len(parts) - 1
            self.col = len(parts[-1])
        self.dirty = True

    # -- keys

    def key(self, k) -> bool:
        """One key; False if it isn't the editor's (so the screen can use it)."""
        self.message = ""
        if self.prompt is not None:
            return self.key_prompt(k)
        shift = k in (S_LEFT, S_RIGHT, S_UP, S_DOWN, S_HOME, S_END)
        moves = {curses.KEY_LEFT: S_LEFT, curses.KEY_RIGHT: S_RIGHT, curses.KEY_UP: S_UP, curses.KEY_DOWN: S_DOWN,
                 curses.KEY_HOME: S_HOME, curses.KEY_END: S_END}
        if shift or k in moves or k in (curses.KEY_PPAGE, curses.KEY_NPAGE):
            if shift and self.anchor is None:
                self.anchor = (self.row, self.col)
            if not shift:
                self.anchor = None
            base = {v: kk for kk, v in moves.items()}.get(k, k)
            self.move(base)
            self.last_kind = ""
            return True
        if k in ("\x13",):                                       # Ctrl-S
            return self.save() or True
        if k == "\x1a":                                          # Ctrl-Z
            self._swap(self.undo, self.redo)
            return True
        if k == "\x19":                                          # Ctrl-Y
            self._swap(self.redo, self.undo)
            return True
        if k in ("\x03", "\x18"):                                # Ctrl-C, Ctrl-X
            sel = self.selected_text()
            if sel:
                self.clip, self.clip_lines = sel, False
                if k == "\x18":
                    self.snapshot("cut")
                    self.delete_selection()
            else:
                self.clip, self.clip_lines = self.lines[self.row] + "\n", True
                if k == "\x18":
                    self.snapshot("cut")
                    self.cut_line()
            self.message = "cut" if k == "\x18" else "copied"
            return True
        if k in ("\x16", "\x15"):                                # Ctrl-V (Ctrl-U too, as nano)
            if not self.clip:
                return True
            self.snapshot("paste")
            if not self.delete_selection() and self.clip_lines:
                self.col = 0
            self.insert(self.clip)
            return True
        if k == "\x0b":                                          # Ctrl-K
            self.snapshot("cut")
            self.anchor = None
            self.cut_line()
            return True
        if k == "\x04":                                          # Ctrl-D
            self.snapshot("dup")
            self.lines.insert(self.row + 1, self.lines[self.row])
            self.row += 1
            self.dirty = True
            return True
        if k == "\x06":                                          # Ctrl-F
            self.prompt, self.answer = "find", self.last_find
            return True
        if k == "\x07":                                          # Ctrl-G
            self.prompt, self.answer = "line", ""
            return True
        if k in (curses.KEY_BACKSPACE, "\x7f", "\x08", 127, 8):
            self.snapshot("del")
            if self.delete_selection():
                return True
            if self.col > 0:
                line = self.lines[self.row]
                # in leading spaces, a whole indent at a time
                n = len(INDENT) if line[:self.col].strip() == "" and self.col % len(INDENT) == 0 else 1
                self.lines[self.row] = line[:self.col - n] + line[self.col:]
                self.col -= n
            elif self.row > 0:
                self.col = len(self.lines[self.row - 1])
                self.lines[self.row - 1:self.row + 1] = [self.lines[self.row - 1] + self.lines[self.row]]
                self.row -= 1
            self.dirty = True
            return True
        if k == curses.KEY_DC:
            self.snapshot("del")
            if self.delete_selection():
                return True
            line = self.lines[self.row]
            if self.col < len(line):
                self.lines[self.row] = line[:self.col] + line[self.col + 1:]
            elif self.row + 1 < len(self.lines):
                self.lines[self.row:self.row + 2] = [line + self.lines[self.row + 1]]
            self.dirty = True
            return True
        if k in ("\n", "\r", curses.KEY_ENTER, 10, 13):
            self.snapshot("enter")
            self.delete_selection()
            line = self.lines[self.row]
            lead = line[:len(line) - len(line.lstrip(" "))][:self.col]
            if line[:self.col].rstrip().endswith("{"):
                lead += INDENT
            self.insert("\n" + lead)
            return True
        if k == "\t":
            self.snapshot("indent")
            sel = self.selection()
            if sel:
                self.indent(sel, +1)
            else:
                self.insert(" " * (len(INDENT) - self.col % len(INDENT)))
            return True
        if k == BTAB:
            self.snapshot("indent")
            sel = self.selection() or ((self.row, 0), (self.row, 0))
            self.indent(sel, -1)
            return True
        if isinstance(k, str) and len(k) == 1 and (k.isprintable() or k == " "):
            self.snapshot("type")
            self.delete_selection()
            self.insert(k)
            if k == "}" and self.lines[self.row][:self.col - 1].strip() == "" and self.col - 1 >= len(INDENT):
                line = self.lines[self.row]           # a closing brace lines up with its opening line
                self.lines[self.row] = line[len(INDENT):]
                self.col -= len(INDENT)
            return True
        return False

    def cut_line(self) -> None:
        if len(self.lines) == 1:
            self.lines = [""]
        else:
            del self.lines[self.row]
            self.row = min(self.row, len(self.lines) - 1)
        self.col = min(self.col, len(self.lines[self.row]))
        self.dirty = True

    def indent(self, sel, way: int) -> None:
        (r0, _), (r1, c1) = sel
        if c1 == 0 and r1 > r0:
            r1 -= 1
        for r in range(r0, r1 + 1):
            line = self.lines[r]
            if way > 0:
                self.lines[r] = INDENT + line
            else:
                n = min(len(INDENT), len(line) - len(line.lstrip(" ")))
                self.lines[r] = line[n:]
                if r == self.row:
                    self.col = max(0, self.col - n)
        if way > 0:
            self.col += len(INDENT)
            if self.anchor:
                self.anchor = (self.anchor[0], self.anchor[1] + len(INDENT) if self.anchor[1] else 0)
        self.col = min(self.col, len(self.lines[self.row]))
        self.dirty = True

    def move(self, k) -> None:
        n = len(self.lines)
        if k == curses.KEY_LEFT:
            if self.col > 0:
                self.col -= 1
            elif self.row > 0:
                self.row -= 1
                self.col = len(self.lines[self.row])
            self.want = self.col
        elif k == curses.KEY_RIGHT:
            if self.col < len(self.lines[self.row]):
                self.col += 1
            elif self.row + 1 < n:
                self.row, self.col = self.row + 1, 0
            self.want = self.col
        elif k in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE, curses.KEY_NPAGE):
            step = {curses.KEY_UP: -1, curses.KEY_DOWN: 1, curses.KEY_PPAGE: -max(1, self.height - 2),
                    curses.KEY_NPAGE: max(1, self.height - 2)}[k]
            self.row = max(0, min(n - 1, self.row + step))
            self.col = min(self.want, len(self.lines[self.row]))
        elif k == curses.KEY_HOME:                   # first to the text's start, then to column 0
            ind = len(self.lines[self.row]) - len(self.lines[self.row].lstrip(" "))
            self.col = ind if self.col != ind else 0
            self.want = self.col
        elif k == curses.KEY_END:
            self.col = len(self.lines[self.row])
            self.want = self.col

    def goto(self, row: int, col: int = 0) -> None:
        self.row = max(0, min(len(self.lines) - 1, row))
        self.col = max(0, min(col, len(self.lines[self.row])))
        self.want = self.col
        self.anchor = None

    def key_prompt(self, k) -> bool:
        """Typing an answer at the bottom: Enter does it, Esc doesn't; in find, Ctrl-F finds the next and stays."""
        if k in ("\x1b", 27, "\x03"):
            self.prompt = None
            return True
        if k in ("\n", "\r", curses.KEY_ENTER, 10, 13, "\x06"):
            what, ans = self.prompt, self.answer
            if k != "\x06" or what != "find":
                self.prompt = None
            if what == "find":
                self.last_find = ans or self.last_find
                self.find_next()
            elif ans.strip().isdigit():
                self.goto(int(ans) - 1)
            return True
        if k in (curses.KEY_BACKSPACE, "\x7f", "\x08", 127, 8):
            self.answer = self.answer[:-1]
            return True
        if isinstance(k, str) and len(k) == 1 and k.isprintable():
            self.answer += k
        return True

    def find_next(self) -> None:
        """The next place the text is found after the cursor, wrapping round; it is selected."""
        needle = self.last_find
        if not needle:
            return
        n = len(self.lines)
        for i in range(n + 1):
            r = (self.row + i) % n
            at = self.lines[r].find(needle, self.col if i == 0 else 0)
            if at >= 0:
                self.anchor = (r, at)
                self.row, self.col = r, at + len(needle)
                self.want = self.col
                return
        self.message = f"not found: {needle}"

    # -- drawing

    def scroll(self, h: int, w: int) -> None:
        self.height = h
        if self.row < self.top:
            self.top = self.row
        elif self.row >= self.top + h:
            self.top = self.row - h + 1
        self.top = max(0, min(self.top, max(0, len(self.lines) - 1)))
        if self.col < self.left:
            self.left = self.col
        elif self.col >= self.left + w - 1:
            self.left = self.col - w + 2

    def draw(self, cv, y: int, x: int, h: int, w: int, col: dict, *, ptx: bool, mapped=(), errs=(),
             focused: bool = True) -> tuple[int, int] | None:
        """The text in h rows by w columns (line numbers included); where the cursor goes on screen."""
        gutter = 6
        tw = max(1, w - gutter)
        rows = h - (1 if self.prompt is not None else 0)
        self.scroll(rows, tw)
        sel = self.selection()
        for i in range(rows):
            r = self.top + i
            if r >= len(self.lines):
                break
            n = r + 1
            mark = "✗" if n in errs else "•" if n in mapped else " "
            cv.put(y + i, x, f"{n:>4}{mark} ", col["bad"] if n in errs else col["key"] if r == self.row
                   else col["dim"])
            line = self.lines[r]
            seg = line[self.left:self.left + tw]
            cv.put(y + i, x + gutter, seg.ljust(tw), col["text"], col["sel"] if r == self.row and focused else None)
            for s, e, c in colour_spans(line, ptx):
                s2, e2 = max(s, self.left), min(e, self.left + tw)
                if s2 < e2:
                    cv.put(y + i, x + gutter + s2 - self.left, line[s2:e2], col[c],
                           col["sel"] if r == self.row and focused else None)
            if sel:
                (r0, c0), (r1, c1) = sel
                if r0 <= r <= r1:
                    a = c0 if r == r0 else 0
                    b = c1 if r == r1 else len(line) + 1
                    a2, b2 = max(a, self.left), min(b, self.left + tw)
                    if a2 < b2:
                        cv.put(y + i, x + gutter + a2 - self.left, (line + " ")[a2:b2].ljust(b2 - a2), col["text"],
                               col["attn"])
        if self.prompt is not None:
            label = "find: " if self.prompt == "find" else "go to line: "
            cv.put(y + h - 1, x, (label + self.answer + "   (Enter, Esc)")[:w].ljust(w), col["key"])
            return (y + h - 1, x + len(label) + len(self.answer))
        return (y + self.row - self.top, x + gutter + self.col - self.left)
