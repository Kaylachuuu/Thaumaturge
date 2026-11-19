"""The lessons, and your own work: where each one's files are, and what you have done so far.

A lesson lives in thaumaturge/bitscryer/lessons/NN-name/:

    lesson.md     what it teaches and what to write
    hints.md      hints, one at a time, only when you ask (separated by lines of ---)
    starter.cu    (or starter.ptx) the file you start from
    check.py      what to run and how to check it (bitscryer/spec.py)

Your copy goes to ~/bitscryer/lessons/NN-name/kernel.cu the first time you open it, and is never written over by
BitScryer after that. Your own projects (anything you write once the lessons are done) are folders anywhere: the
kernel file, and a check.py if you want it run and checked (`bitscryer new NAME` starts one).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
LESSONS = HERE / "lessons"
TEMPLATE = HERE / "template"


def home() -> Path:
    return Path(os.environ.get("BITSCRYER_HOME") or Path.home() / "bitscryer")


@dataclass
class Work:
    """One thing open in BitScryer: a lesson or a project of yours."""
    key: str                     # "01-add-lists", or the project's folder
    title: str
    folder: Path                 # your folder
    source: Path                 # your kernel file
    check: Path | None           # check.py, if there is one
    lesson: Path | None = None   # the lesson's own folder (lesson.md, hints.md), None for a project
    needs: tuple = ()

    @property
    def number(self) -> str:
        return self.key[:2] if self.lesson else ""

    def text(self) -> str:
        if not self.lesson:
            readme = self.folder / "README.md"
            return readme.read_text() if readme.exists() else (
                "Your own project. BitScryer builds your file every time you save it, and shows the PTX and SASS.\n\n"
                "If there is a check.py beside it, it is run on the card and checked too (see the one `bitscryer "
                "new` writes, or any lesson's check.py, for how they're written).")
        return (self.lesson / "lesson.md").read_text()

    def hints(self) -> list[str]:
        if not self.lesson or not (self.lesson / "hints.md").exists():
            return []
        parts, cur = [], []
        for line in (self.lesson / "hints.md").read_text().splitlines():
            if line.strip() == "---":
                parts.append("\n".join(cur).strip())
                cur = []
            else:
                cur.append(line)
        parts.append("\n".join(cur).strip())
        return [p for p in parts if p]


def _title(check: Path, fallback: str) -> tuple[str, str, tuple]:
    """TITLE, SOURCE and NEEDS read from a check.py without running it (simple assignments only)."""
    import ast
    title, source, needs = fallback, "kernel.cu", ()
    try:
        tree = ast.parse(check.read_text())
    except (OSError, SyntaxError):
        return title, source, needs
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                v = ast.literal_eval(node.value)
            except ValueError:
                continue
            name = node.targets[0].id
            if name == "TITLE":
                title = v
            elif name == "SOURCE":
                source = v
            elif name == "NEEDS":
                needs = tuple(v)
    return title, source, needs


def lessons(base: Path | None = None) -> list[Work]:
    base = base or home()
    out = []
    for d in sorted(p for p in LESSONS.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        title, source, needs = _title(d / "check.py", d.name)
        mine = base / "lessons" / d.name
        out.append(Work(d.name, title, mine, mine / source, d / "check.py", d, needs))
    return out


def open_lesson(w: Work) -> None:
    """Your copy of the starter file, made the first time; never written over."""
    w.folder.mkdir(parents=True, exist_ok=True)
    if not w.source.exists():
        starter = w.lesson / ("starter" + w.source.suffix)
        shutil.copyfile(starter, w.source)


def project(folder: Path) -> Work:
    """A folder of yours (or a single file): the kernel file in it, and check.py if there is one."""
    folder = folder.expanduser().resolve()
    if folder.is_file():
        src, folder = folder, folder.parent
    else:
        check = folder / "check.py"
        _, named, _ = _title(check, "") if check.exists() else ("", "", ())
        src = folder / named if named and (folder / named).exists() else None
        if src is None:
            found = sorted([*folder.glob("*.cu"), *folder.glob("*.ptx")])
            src = found[0] if found else folder / "kernel.cu"
    check = folder / "check.py"
    title, _, needs = _title(check, folder.name) if check.exists() else (folder.name, "", ())
    return Work(str(folder), title, folder, src, check if check.exists() else None, None, needs)


def new_project(folder: Path) -> Work:
    """A new project folder with a kernel and a check.py to change: the start of your own work."""
    folder = folder.expanduser().resolve()
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"{folder} already has files in it")
    folder.mkdir(parents=True, exist_ok=True)
    for f in TEMPLATE.iterdir():
        shutil.copyfile(f, folder / f.name)
    return project(folder)


class Progress:
    """What you have done: lessons passed, hints seen, the last thing open, the card you chose. One small JSON file."""

    def __init__(self, path: Path | None = None):
        self.path = path or home() / "progress.json"
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}
        self.data.setdefault("passed", [])
        self.data.setdefault("hints", {})

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1))
            tmp.replace(self.path)
        except OSError:
            pass

    def passed(self, key: str) -> bool:
        return key in self.data["passed"]

    def mark_passed(self, key: str) -> None:
        if key not in self.data["passed"]:
            self.data["passed"].append(key)
            self.save()

    def hints_seen(self, key: str) -> int:
        return int(self.data["hints"].get(key, 0))

    def see_hint(self, key: str, n: int) -> None:
        self.data["hints"][key] = max(n, self.hints_seen(key))
        self.save()

    def get(self, k: str, default=None):
        return self.data.get(k, default)

    def set(self, k: str, v) -> None:
        self.data[k] = v
        self.save()
