"""Builds a model from its stack file and blocks, on the CPU, and checks it with a tiny pass:

1. the stack file reads,
2. every block's file loads and has one block in it,
3. the plugs fit, top to bottom (what each block gives is what the next one takes),
4. every block can be made with the settings it asks for,
5. a few words go all the way through, and what comes out is the shape it should be,

and it counts the weights and works out the card memory, in fp16, to run it and (roughly) to train it.

It runs the blocks' own code, so the screen runs it in a program of its own, on the CPU only, with a time limit:

    python -m thaumaturge.artificer.check FOLDER [--json]
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import traceback
from pathlib import Path

from . import stack as S

HERE = Path(__file__).resolve().parent
LIBRARY = HERE / "blocks"
CHECK_WORDS = 16                       # words in the tiny pass
TRAIN_BATCH = 8                        # what "to train" is worked out for, unless the stack says batch = N
SPECIAL = {"repeats"}                  # settings the builder gives every block that asks


def where(folder: Path) -> tuple[Path, Path]:
    """A model's stack file and its blocks folder."""
    stacks = sorted(folder.glob("*.stack"))
    return (stacks[0] if stacks else folder / "model.stack"), folder / "blocks"


def block_file(blocks: Path, name: str) -> Path:
    return blocks / f"{name}.py"


def key(path: tuple) -> str:
    return ".".join(str(i) for i in path)


class Problem(Exception):
    def __init__(self, path, msg, file=None, line=0):
        super().__init__(msg)
        self.path, self.msg, self.file, self.line = path, msg, file, line


def blame(e: BaseException, blocks: Path) -> tuple[str | None, int]:
    """The block file and line an error came from, if it came from one."""
    for fr in reversed(traceback.extract_tb(e.__traceback__)):
        try:
            p = Path(fr.filename).resolve()
        except OSError:
            continue
        if p.parent == blocks.resolve():
            return p.name, fr.lineno or 0
    if isinstance(e, SyntaxError) and e.filename and Path(e.filename).parent.resolve() == blocks.resolve():
        return Path(e.filename).name, e.lineno or 0
    return None, 0


def load_class(path: Path):
    """The one Block class a block file defines."""
    from .block import Block
    spec = importlib.util.spec_from_file_location(f"artificer_block_{path.stem.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    found = [v for v in vars(mod).values()
             if inspect.isclass(v) and issubclass(v, Block) and v is not Block and v.__module__ == mod.__name__]
    if len(found) != 1:
        raise ValueError(f"{path.name} should define one block (a class built on Block); it defines {len(found)}")
    cls = found[0]
    for side in ("takes", "gives"):
        from .block import PLUGS
        if getattr(cls, side) not in PLUGS:
            raise ValueError(f"{cls.__name__}.{side} is {getattr(cls, side)!r}; a plug is one of {', '.join(PLUGS)}")
    return cls


def settings_for(cls, top: dict, own: dict, repeats: int, name: str) -> tuple[dict, list[str]]:
    """What the block's setup() asks for, from its own line, else the top of the stack; and what's missing."""
    sig = inspect.signature(cls.setup)
    give, missing = {}, []
    for p in list(sig.parameters.values())[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in own:
            give[p.name] = own[p.name]
        elif p.name == "repeats":
            give[p.name] = repeats
        elif p.name in top:
            give[p.name] = top[p.name]
        elif p.default is p.empty:
            missing.append(p.name)
    return give, missing


def plugs(st: S.Stack, classes: dict) -> list[Problem]:
    """Walk the stack top to bottom, carrying what the last block gave; a problem wherever it doesn't fit."""
    problems = []

    def seq(nodes, path, have):
        for i, nd in enumerate(nodes):
            p = path + (i,)
            if nd.kind == "block":
                cls = classes.get(nd.name)
                if cls is None:
                    have = None                     # can't tell; carry on without complaining further
                    continue
                if have is not None and cls.takes != have:
                    problems.append(Problem(p, f"{nd.name} takes {cls.takes}, but what comes before gives {have}"))
                have = cls.gives
            elif nd.kind == "add":
                if have not in (None, "width"):
                    problems.append(Problem(p, f"a shortcut adds rows of width; here comes {have}"))
                out = seq(nd.children, p, "width")
                if out not in (None, "width"):
                    problems.append(Problem(p, f"inside a shortcut the blocks must end with width, not {out}"))
                have = "width" if have is not None else None
            else:
                out = seq(nd.children, p, have)
                if nd.count > 1 and out != have and None not in (out, have):
                    problems.append(Problem(p, f"repeated, it must give what it takes ({have}), but gives {out}"))
                have = out
        return have
    end = seq(st.nodes, (), "tokens")
    if end not in (None, "logits"):
        problems.append(Problem((len(st.nodes) - 1,) if st.nodes else (), f"the model ends giving {end}; a model "
                                f"ends giving logits (an out block)"))
    return problems


def build(st: S.Stack, classes: dict, blocks: Path):
    """The model (a torch Module) and, by path, the module made for each node."""
    import torch.nn as nn

    class Shortcut(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

    made: dict[tuple, object] = {}
    copies: dict[tuple, list] = {}          # a block inside a repeat: every copy of it
    firsts: dict[str, object] = {}
    problems: list[Problem] = []

    def make(nodes, path, repeats, copy_no):
        mods = []
        for i, nd in enumerate(nodes):
            p = path + (i,)
            if nd.kind == "block":
                cls = classes[nd.name]
                give, missing = settings_for(cls, st.settings, nd.settings, repeats, nd.name)
                if missing:
                    raise Problem(p, f"{nd.name} needs {', '.join(missing)}: set it at the top "
                                     f"({missing[0]} = ...) or on the block ({nd.name} {missing[0]}=...)")
                try:
                    m = cls(**give)
                except Problem:
                    raise
                except Exception as e:  # noqa: BLE001 (the block's own code)
                    f, ln = blame(e, blocks)
                    raise Problem(p, f"{nd.name} couldn't be made: {type(e).__name__}: {e}", f, ln) from e
                m._artificer = (p, nd.name)
                firsts.setdefault(nd.name, m)
                if copy_no == 0:
                    made[p] = m
                copies.setdefault(p, []).append(m)
                mods.append(m)
            elif nd.kind == "add":
                inner = nn.ModuleList(make(nd.children, p, repeats, copy_no))
                s = Shortcut(inner)
                s._artificer = (p, "add")
                if copy_no == 0:
                    made[p] = s
                mods.append(s)
            else:
                for c in range(nd.count):
                    mods.extend(make(nd.children, p, repeats * nd.count, copy_no if c == 0 else c + copy_no))
        return mods
    top = make(st.nodes, (), 1, 0)
    for m in top_modules(top):
        if hasattr(m, "borrow"):
            try:
                m.borrow(firsts)
            except Exception as e:  # noqa: BLE001
                f, ln = blame(e, blocks)
                problems.append(Problem(m._artificer[0], f"{m._artificer[1]}: {e}", f, ln))
    model = nn.ModuleList(top)
    return model, made, copies, problems


def top_modules(mods):
    for m in mods:
        yield m
        inner = getattr(m, "inner", None)
        if inner is not None:
            yield from top_modules(inner)


def run(mods, x, shapes: dict, blocks: Path):
    """Through the modules in order; a shortcut adds what its blocks make to what went in. Each node's output
    shape kept (the first copy's), and its size in numbers (all copies)."""
    for m in mods:
        p, name = m._artificer
        inner = getattr(m, "inner", None)
        try:
            if inner is not None:
                y = x + run(inner, x, shapes, blocks)
            else:
                y = m(x)
        except Problem:
            raise
        except Exception as e:  # noqa: BLE001
            f, ln = blame(e, blocks)
            got = tuple(x.shape)
            raise Problem(p, f"{name}, given {got}: {type(e).__name__}: {e}", f, ln) from e
        if inner is None and getattr(m, "takes", "") == "width" == getattr(m, "gives", "") \
                and tuple(y.shape) != tuple(x.shape):
            raise Problem(p, f"{name} took {tuple(x.shape)} and gave {tuple(y.shape)}: a block that takes and "
                             f"gives width gives back the same shape")
        shapes.setdefault(key(p), tuple(y.shape))
        shapes["_numbers"] = shapes.get("_numbers", 0) + y.numel()
        x = y
    return x


def check(folder: Path) -> dict:
    folder = Path(folder)
    stack_path, blocks = where(folder)
    out = {"stack": stack_path.name, "ok": False, "problems": [], "nodes": {}, "totals": {}}

    def problem(p: Problem | None = None, *, path=(), msg="", file=None, line=0):
        if p is not None:
            path, msg, file, line = p.path, p.msg, p.file, p.line
        out["problems"].append({"path": key(path) if path else "", "msg": msg, "file": file, "line": line})

    try:
        text = stack_path.read_text()
    except OSError as e:
        problem(msg=f"can't read {stack_path}: {e}")
        return out
    st = S.parse(text)
    for ln, msg in st.errors:
        problem(msg=msg, file=stack_path.name, line=ln)
    if st.errors:
        return out

    classes, docs = {}, {}
    rows = S.rows(st)
    for p, _, nd in rows:
        if nd.kind != "block" or nd.name in classes or nd.name in docs:
            continue
        f = block_file(blocks, nd.name)
        if not f.exists():
            there = (LIBRARY / f.name).exists()
            problem(path=p, msg=f"no blocks/{f.name}" + (" (it is in the library: remove the line and add it with "
                                                       "the add key)" if there else ""))
            docs[nd.name] = None
            continue
        try:
            classes[nd.name] = load_class(f)
        except Exception as e:  # noqa: BLE001
            fl, ln = blame(e, blocks)
            what = f"{e.msg}" if isinstance(e, SyntaxError) else f"{type(e).__name__}: {e}"
            problem(path=p, msg=f"{f.name}: {what}", file=fl or f.name, line=ln)
            docs[nd.name] = None
    for p in plugs(st, classes):
        problem(p)
    for p, _, nd in rows:
        info = {}
        if nd.kind == "block" and nd.name in classes:
            info = {"takes": classes[nd.name].takes, "gives": classes[nd.name].gives}
        out["nodes"][key(p)] = info
    if out["problems"]:
        return out

    import torch
    torch.manual_seed(0)
    try:
        model, made, copies, more = build(st, classes, blocks)
    except Problem as p:
        problem(p)
        return out
    for p in more:
        problem(p)
    if out["problems"]:
        return out

    # weights: each node's own (all its copies, for a block in a repeat), and the whole model's, shared ones once
    def own(m):
        return {id(t): t.numel() for t in m.parameters()}
    for p, _, nd in rows:
        k = key(p)
        if nd.kind == "block" and p in copies:
            ids = {}
            for c in copies[p]:
                ids.update(own(c))
            out["nodes"][k]["weights"] = sum(ids.values())
            out["nodes"][k]["each"] = sum(own(copies[p][0]).values())
            out["nodes"][k]["copies"] = len(copies[p])
            sh = copies[p][0].shares()
            if sh:
                out["nodes"][k]["shares"] = sh
    for p, _, nd in rows:                    # a group's weights: everything under it
        if nd.group:
            out["nodes"][key(p)]["weights"] = sum(out["nodes"][key(q)].get("weights", 0)
                                                  for q, _, n2 in rows if n2.kind == "block" and q[:len(p)] == p)
    total = sum({id(t): t.numel() for t in model.parameters()}.values())

    vocab = st.settings.get("vocab", 256)
    words = min(CHECK_WORDS, int(st.settings.get("context", CHECK_WORDS)))
    x = torch.randint(0, int(vocab) if isinstance(vocab, int) and vocab > 0 else 256, (1, words))
    shapes: dict = {}
    try:
        with torch.no_grad():
            y = run(list(model), x, shapes, blocks)
    except Problem as p:
        problem(p)
        return out
    for k, shape in shapes.items():
        if k in out["nodes"]:
            out["nodes"][k]["shape"] = list(shape)
    want = (1, words, vocab) if isinstance(vocab, int) else None
    if want and tuple(y.shape) != want:
        problem(msg=f"what comes out is {tuple(y.shape)}; for {words} words it should be {want}")
    if not torch.isfinite(y).all():
        problem(msg="what comes out has inf or NaN in it, even in fp32 on the CPU")

    context = int(st.settings.get("context", 512))
    batch = int(st.settings.get("batch", TRAIN_BATCH))
    per_word = shapes.get("_numbers", 0) / words                 # numbers each word leaves behind, all blocks
    activations = per_word * context * batch * 2 * 3             # fp16, and blocks keep about 3x inside (rough)
    out["totals"] = {
        "weights": total,
        "run_bytes": total * 2,                                  # fp16 weights
        "train_bytes": total * 16 + activations,                 # fp16 + fp32 copy, grads, Adam's two: ~16 a weight
        "train_batch": batch,
        "context": context,
        "words": words,
    }
    out["ok"] = not out["problems"]
    return out


def size(n: float) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n:.0f} B"


def count(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def report(res: dict) -> str:
    lines = []
    if res["ok"]:
        t = res["totals"]
        lines.append(f"✓ everything fits: {count(t['weights'])} weights")
        lines.append(f"  to run, in fp16: {size(t['run_bytes'])}")
        lines.append(f"  to train, batch {t['train_batch']} × {t['context']} words: about {size(t['train_bytes'])}")
    for p in res["problems"]:
        where_ = f"{p['file']}:{p['line']} " if p.get("file") else ""
        lines.append(f"✗ {where_}{p['msg']}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="artificer-check", description="Build a model on the CPU and check it.")
    ap.add_argument("folder")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        res = check(Path(a.folder))
    except Exception as e:  # noqa: BLE001  (never a traceback in place of an answer)
        res = {"ok": False, "problems": [{"path": "", "msg": f"Artificer itself failed: {type(e).__name__}: {e}",
                                          "file": None, "line": 0}], "nodes": {}, "totals": {}}
    print(json.dumps(res) if a.json else report(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
