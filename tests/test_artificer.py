#!/usr/bin/env python3
"""Checks Artificer (thaumaturge/artificer/) without a card.

    python tests/test_artificer.py

1. Stack files: read and written back unchanged; mistakes found with their lines; every change the screen makes
   (add, remove, copy, move in and out of groups, shortcut, repeat, ungroup, settings).
2. Block files described without running them (plugs, first line).
3. If PyTorch is here: the starter model checks out (123.6M weights), and each kind of mistake is found and put
   on the right block: plugs that don't fit, a missing setting, a block that changes the width, a mistake in a
   block's code (with its line), a block stuck in a loop is stopped.
4. The screen's keys, without a terminal: add from the library, a new block, move, shortcut, repeat, settings,
   remove, undo; editing a block and saving it; drawn as text.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ARTIFICER_AS_ROOT"] = "1"
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp()

from thaumaturge.artificer import check as C  # noqa: E402
from thaumaturge.artificer import screen as SC  # noqa: E402
from thaumaturge.artificer import stack as S  # noqa: E402
from thaumaturge.common.screen import TextCanvas  # noqa: E402

TORCH = importlib.util.find_spec("torch") is not None


def check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(what)
    print(f"ok  {what}")


STARTER = (SC.STARTERS / "tiny-story.stack").read_text()


def labels(st) -> list[str]:
    return ["  " * d + nd.label() for _, d, nd in S.rows(st)]


def test_stack() -> None:
    st = S.parse(STARTER)
    check(not st.errors and S.write(st) == STARTER, "the starter stack reads and is written back unchanged")
    check(st.settings == {"width": 768, "heads": 12, "vocab": 50304, "context": 512}, "settings at the top read")
    check(labels(st)[:4] == ["embed", "repeat ×12", "  add (shortcut)", "    norm"], "groups by indenting")
    bad = S.parse("width = 8\n\nembed\nrepeat x\n    norm\nadd\nnorm\n    out bad\n")
    lines = {ln for ln, _ in bad.errors}
    check({4, 6, 8} <= lines, f"mistakes found on their lines: {bad.errors}")

    st = S.parse("embed\nrepeat 2\n    add\n        norm\n        attention\nnorm\nout\n")
    p = S.insert(st, (0,), S.Node("block", name="norm"))
    check(p == (1,) and labels(st)[1] == "norm", "add after the chosen block")
    p = S.insert(st, (2,), S.Node("block", name="x"))
    check(p == (2, 0) and S.node(st, p).name == "x", "add on a group goes in at its top")
    S.remove(st, p)
    p = S.move(st, (1,), +1)                                   # norm next to the repeat: steps into it
    check(p == (1, 0) and S.node(st, p).name == "norm", "moving down into the group below")
    p = S.move(st, p, -1)                                      # at a group's top: steps out, above it
    check(p == (1,) and S.node(st, p).name == "norm", "moving up out of a group")
    p = S.move(st, (0,), -1)
    check(p == (0,), "the very top stays put")
    st2 = S.parse("a\nadd\n    b\nc\n")
    p = S.move(st2, (1, 0), +1)
    check(labels(st2) == ["a", "b", "c"] and p == (1,), "a group left empty goes away")
    p = S.wrap(st, (0,), "add")
    check(S.node(st, p).kind == "add" and S.node(st, p + (0,)).name == "embed", "put inside a shortcut")
    S.unwrap(st, p)
    check(S.node(st, (0,)).name == "embed", "the shortcut taken away again")
    p = S.wrap(st, (1,), "repeat", 3)
    check(S.node(st, p).count == 3, "repeat it 3 times")
    S.unwrap(st, p)
    p = S.duplicate(st, (0,))
    check(S.node(st, p).name == "embed" and len(st.nodes) == 6, "copy")
    nd = S.node(st, (0,))
    check(S.set_settings(nd, "heads=8 eps=1e-5") == [] and nd.settings == {"heads": 8, "eps": 1e-5},
          "a block's own settings")
    check(S.set_settings(nd, "heads=") == [] and nd.settings == {"eps": 1e-5}, "a setting taken away")
    check(S.set_settings(nd, "oops") != [], "a setting without = is refused")
    text = S.write(st)
    check(S.write(S.parse(text)) == text and "embed eps=1e-05" in text, "written with its settings, reads back")


def test_describe() -> None:
    d = SC.describe(C.LIBRARY / "attention.py")
    check(d["takes"] == "width" and d["gives"] == "width" and d["about"].startswith("Each word looks back"),
          "a block described without running it")
    names = [b["name"] for b in SC.available(Path(tempfile.mkdtemp()))]
    check("template" not in names and {"embed", "norm", "attention", "feed-forward", "out"} <= set(names),
          "the library offers every block but the template")


def model(text: str = STARTER) -> Path:
    folder = Path(tempfile.mkdtemp()) / "m"
    SC.new_model(folder)
    (folder / "m.stack").write_text(text)
    return folder


def test_check() -> None:
    if not TORCH:
        print("--  PyTorch isn't here: the checks that build a model are skipped")
        return
    folder = model()
    res = C.check(folder)
    check(res["ok"] and res["totals"]["weights"] == 123_587_328, "the starter model: 123.6M weights, fits")
    check(res["nodes"]["3"].get("shares") == "embed", "out uses the embedding's table")
    check(res["nodes"]["1.0.1"]["copies"] == 12 and res["nodes"]["1.0.1"]["shape"] == [1, 16, 768],
          "a repeated block: its copies and its shape")

    def problems(text, edit=None):
        f = model(text)
        if edit:
            edit(f)
        r = C.check(f)
        return [(p["path"], p["msg"], p.get("file"), p.get("line")) for p in r["problems"]]
    top = "width = 64\nheads = 4\nvocab = 100\n\n"
    got = problems(top + "embed\nout\nnorm\n")
    check(any(p == "2" and "takes width" in m for p, m, _, _ in got), "plugs that don't fit, on the block")
    got = problems("width = 64\nvocab = 100\n\nembed\nattention\nout\n")
    check(any(p == "1" and "needs heads" in m for p, m, _, _ in got), "a missing setting")

    def wider(f):
        b = f / "blocks" / "feed-forward.py"
        b.write_text(b.read_text().replace("wider * width, width,", "wider * width, width + 1,"))
    got = problems(top + "embed\nadd\n    feed-forward\nout\n", wider)
    check(any(p == "1.0" and "same shape" in m for p, m, _, _ in got), "a block that changes the width")

    def broken(f):
        b = f / "blocks" / "norm.py"
        b.write_text(b.read_text().replace("return (xf * rms)", "return (xf * nothing)"))
    got = problems(top + "embed\nnorm\nout\n", broken)
    check(any(p == "1" and fl == "norm.py" and ln and "nothing" in m for p, m, fl, ln in got),
          f"a mistake in a block's code, with its file and line: {got}")

    def syntax(f):
        (f / "blocks" / "norm.py").write_text("class (:\n")
    got = problems(top + "embed\nnorm\nout\n", syntax)
    check(any(fl == "norm.py" and ln == 1 for _, _, fl, ln in got), "a block that isn't Python, with its line")

    f = model(top + "embed\nslow\nout\n")
    (f / "blocks" / "slow.py").write_text((C.LIBRARY / "template.py").read_text().replace(
        "class NAME(", "class Slow(").replace("return self.layer(x)", "while True:\n            pass"))
    old = SC.CHECK_LIMIT
    SC.CHECK_LIMIT = 5
    res = SC.run_check(f)
    SC.CHECK_LIMIT = old
    check(not res["ok"] and "stopped" in res["problems"][0]["msg"], "a block stuck in a loop is stopped")


def test_screen() -> None:
    folder = model("width = 64\nheads = 4\nvocab = 100\n\nembed\nnorm\nout\n")
    (folder / "blocks" / "feed-forward.py").unlink()
    app = SC.App(folder, check_now=TORCH)
    app.wait()
    check(app.editing and app.editing.name == "embed.py", "opens on the first block's code")
    app.key("a")
    names = [c["name"] for c in app.choices]
    i = names.index("feed-forward")
    check(app.choices[i]["where"] == "library", "a block not in the model yet comes from the library")
    for _ in range(i):
        app.key(SC.curses.KEY_DOWN)
    app.key("\n")
    check((folder / "blocks" / "feed-forward.py").exists() and "feed-forward" in labels(app.st)[1],
          "added after the chosen block, its file copied in")
    app.key("g")
    check(labels(app.st)[1:3] == ["add (shortcut)", "  feed-forward"], "g puts it inside a shortcut")
    app.key("r")
    for ch in "\x7f3\n":
        app.key(ch)
    check(app.st.nodes[1].kind == "repeat" and app.st.nodes[1].count == 3, "r repeats it, asking how many times")
    app.key("z")
    check(app.st.nodes[1].kind == "add", "z undoes")
    app.key("]")
    check(labels(app.st)[1:3] == ["norm", "add (shortcut)"], "] moves it down")
    app.key("[")
    app.sel = 0
    app.key("=")
    for ch in "x=2\n":
        app.key(ch)
    check(app.st.nodes[0].settings == {"x": 2} and "embed x=2" in (folder / "m.stack").read_text(),
          "= sets a block's own settings, written to the stack file")
    app.key("d")
    app.key("y")
    check(labels(app.st)[0] == "add (shortcut)", "d removes, after y")
    app.key("z")
    app.key("z")
    check(labels(app.st)[0] == "embed" and not app.st.nodes[0].settings, "undo, twice")
    app.key("n")
    for ch in "my-norm\n":
        app.key(ch)
    f = folder / "blocks" / "my-norm.py"
    check(f.exists() and "class MyNorm(Block)" in f.read_text() and app.focus == "code" and app.editing == f,
          "n makes a block from the template and opens it")
    app.key("#")
    app.key("\x13")
    check(f.read_text().startswith("#") and not app.editor.dirty, "Ctrl-S saves the block")
    app.key("\x1b")
    check(app.focus == "stack", "Esc goes back to the stack")
    app.wait()
    cv = TextCanvas(30, 130)
    app.draw(cv)
    text = cv.text()
    check("the stack · m.stack" in text and "my-norm" in text and "blocks/my-norm.py" in text,
          "drawn as text: the stack and the code")
    if TORCH:
        # the "#" typed above made the first line (the import) a comment: the check finds it, in my-norm.py
        check("✗ 1 problem" in text.split("\n")[0] and "my-norm.py" in app.problems()[0]["msg"],
              f"the check's verdict on top, and the problem in the stack's pane: {app.problems()}")
    app.key("?")
    cv = TextCanvas(40, 130)
    app.draw(cv)
    check("is a shortcut" in cv.text(), "? shows the help")
    check(app.key("\x11") is False, "Ctrl-Q quits")
    shutil.rmtree(folder.parent, ignore_errors=True)


if __name__ == "__main__":
    test_stack()
    test_describe()
    test_check()
    test_screen()
    print("all good")
