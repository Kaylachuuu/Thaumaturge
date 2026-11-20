"""A stack file: a model written as its blocks, in order. No PyTorch in here, so the screen can read and change a
stack without loading any block's code.

    # the small story model                    a comment
    width = 768                                 settings every block may take (by name, as setup() asks for them)
    heads = 12

    embed                                       a block: its file is blocks/embed.py
    repeat 12                                   the blocks indented under it, 12 times over, each with its own weights
        add                                     a shortcut: what goes in comes out again, plus what the blocks
            norm                                    indented under it make of it
            attention heads=8                   a setting for this block alone
        add
            norm
            feed-forward
    norm
    out

Indent with four spaces. A stack is changed by the screen through the functions below, and written back with
`write`, so the file always stays readable and editable by hand.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

INDENT = "    "
GROUPS = ("repeat", "add")


@dataclass
class Node:
    kind: str                                   # "block", "repeat" or "add"
    name: str = ""                              # a block's name (its file is blocks/NAME.py)
    settings: dict = field(default_factory=dict)
    count: int = 1                              # a repeat's
    children: list = field(default_factory=list)
    line: int = 0                               # where it was in the file (counted from 1)

    @property
    def group(self) -> bool:
        return self.kind in GROUPS

    def label(self) -> str:
        if self.kind == "repeat":
            return f"repeat ×{self.count}"
        if self.kind == "add":
            return "add (shortcut)"
        return self.name


@dataclass
class Stack:
    settings: dict = field(default_factory=dict)
    nodes: list = field(default_factory=list)
    comments: list = field(default_factory=list)    # the comment lines at the top, kept when written back
    errors: list = field(default_factory=list)      # (line, what's wrong) from reading it


def value(s: str):
    """A setting's value: a whole number, a decimal, or else the word as written."""
    for kind in (int, float):
        try:
            return kind(s)
        except ValueError:
            pass
    return s


def show(v) -> str:
    return repr(v) if isinstance(v, float) else str(v)


def parse(text: str) -> Stack:
    st = Stack()
    parents: list[tuple[int, list]] = [(-1, st.nodes)]       # (indent, the list new lines at that depth go in)
    seen_block = False
    last = None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            if raw.lstrip().startswith("#") and not seen_block and not st.settings:
                st.comments.append(raw.strip())
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        words = line.split()
        if "=" in line and not seen_block and indent == 0 and len(line.split("=")) == 2 and " " not in \
                line.split("=")[0].strip():
            k, v = (s.strip() for s in line.split("="))
            st.settings[k] = value(v)
            continue
        seen_block = True
        while parents[-1][0] >= indent:
            parents.pop()
        if not parents:
            parents = [(-1, st.nodes)]
        into = parents[-1][1]
        if last and indent > last[0] and not last[1].group:
            st.errors.append((n, f"indented under {last[1].name}, which is a block, not a group (repeat or add)"))
        head = words[0]
        if head == "repeat":
            try:
                count = int(words[1]) if len(words) > 1 else 0
            except ValueError:
                count = 0
            if count < 1:
                st.errors.append((n, "repeat needs a count: repeat 12"))
                count = 1
            node = Node("repeat", count=count, line=n)
        elif head == "add":
            node = Node("add", line=n)
        else:
            settings = {}
            for w in words[1:]:
                if "=" not in w:
                    st.errors.append((n, f"{w}: a block's settings are name=value"))
                    continue
                k, v = w.split("=", 1)
                settings[k] = value(v)
            node = Node("block", name=head, settings=settings, line=n)
        into.append(node)
        last = (indent, node)
        if node.group:
            parents.append((indent, node.children))
    for node in walk(st.nodes):
        if node.group and not node.children:
            st.errors.append((node.line, f"{node.label()} has nothing indented under it"))
    if not st.nodes:
        st.errors.append((0, "no blocks yet"))
    return st


def write(st: Stack) -> str:
    out = list(st.comments)
    for k, v in st.settings.items():
        out.append(f"{k} = {show(v)}")
    if out:
        out.append("")

    def put(nodes, depth):
        for nd in nodes:
            pad = INDENT * depth
            if nd.kind == "repeat":
                out.append(f"{pad}repeat {nd.count}")
            elif nd.kind == "add":
                out.append(f"{pad}add")
            else:
                out.append(pad + " ".join([nd.name] + [f"{k}={show(v)}" for k, v in nd.settings.items()]))
            if nd.group:
                put(nd.children, depth + 1)
    put(st.nodes, 0)
    return "\n".join(out) + "\n"


def walk(nodes):
    for nd in nodes:
        yield nd
        if nd.group:
            yield from walk(nd.children)


def rows(st: Stack) -> list[tuple[tuple, int, Node]]:
    """Every node as the screen lists it, top to bottom: (path, depth, node). A path is the node's place, as a
    tuple of indexes: (1, 0) is the first thing inside the second."""
    out = []

    def go(nodes, path, depth):
        for i, nd in enumerate(nodes):
            p = path + (i,)
            out.append((p, depth, nd))
            if nd.group:
                go(nd.children, p, depth + 1)
    go(st.nodes, (), 0)
    return out


def at(st: Stack, path: tuple) -> tuple[list, int]:
    """The list holding the node at `path`, and its index in it."""
    nodes = st.nodes
    for i in path[:-1]:
        nodes = nodes[i].children
    return nodes, path[-1]


def node(st: Stack, path: tuple) -> Node:
    nodes, i = at(st, path)
    return nodes[i]


# -- changes; each gives the path to select afterwards

def insert(st: Stack, path: tuple | None, nd: Node) -> tuple:
    """`nd` just after the node at `path` (into a group's top, if `path` is a group); at the end if no path."""
    if not path:
        st.nodes.append(nd)
        return (len(st.nodes) - 1,)
    here = node(st, path)
    if here.group:
        here.children.insert(0, nd)
        return path + (0,)
    nodes, i = at(st, path)
    nodes.insert(i + 1, nd)
    return path[:-1] + (i + 1,)


def remove(st: Stack, path: tuple) -> tuple | None:
    nodes, i = at(st, path)
    del nodes[i]
    if nodes:
        return path[:-1] + (min(i, len(nodes) - 1),)
    return path[:-1] or None


def duplicate(st: Stack, path: tuple) -> tuple:
    nodes, i = at(st, path)
    nodes.insert(i + 1, copy.deepcopy(nodes[i]))
    return path[:-1] + (i + 1,)


def move(st: Stack, path: tuple, way: int) -> tuple:
    """One place up (way -1) or down (+1). At the top or bottom of a group it steps out of the group; next to a
    group, it steps into it, so a block can go anywhere by moving it a step at a time."""
    nodes, i = at(st, path)
    nd = nodes[i]
    j = i + way
    if 0 <= j < len(nodes):
        other = nodes[j]
        if other.group:                                          # into the group beside it
            del nodes[i]
            if way > 0:
                other.children.insert(0, nd)
                return path[:-1] + (i, 0)
            other.children.append(nd)
            return path[:-1] + (j, len(other.children) - 1)
        nodes[i], nodes[j] = nodes[j], nodes[i]
        return path[:-1] + (j,)
    if len(path) == 1:                                           # already at the very top or bottom
        return path
    del nodes[i]                                                 # out of its group, before or after it
    outer, g = at(st, path[:-1])
    pos = g if way < 0 else g + 1
    outer.insert(pos, nd)
    if not nodes:                                                # the group it left is empty now: gone
        empty = g if way > 0 else g + 1
        del outer[empty]
        pos = pos if way < 0 else pos - 1
    return path[:-2] + (pos,)


def wrap(st: Stack, path: tuple, kind: str, count: int = 2) -> tuple:
    """The node at `path` put inside a new group ("add", or "repeat" `count` times)."""
    nodes, i = at(st, path)
    nodes[i] = Node(kind, count=count if kind == "repeat" else 1, children=[nodes[i]])
    return path


def unwrap(st: Stack, path: tuple) -> tuple:
    """A group's blocks taken out of it, the group gone."""
    nodes, i = at(st, path)
    g = nodes[i]
    if not g.group:
        return path
    nodes[i:i + 1] = g.children
    return path


def set_settings(nd: Node, text: str) -> list[str]:
    """A block's own settings from `heads=8 width=512` (a name with nothing after = takes the setting away); a
    repeat's count from a number. What couldn't be read."""
    bad = []
    if nd.kind == "repeat":
        try:
            n = int(text.strip())
            if n < 1:
                raise ValueError
            nd.count = n
        except ValueError:
            bad.append(f"{text.strip()}: a repeat's count is a whole number, 1 or more")
        return bad
    for w in text.split():
        if "=" not in w:
            bad.append(f"{w}: settings are name=value")
            continue
        k, v = w.split("=", 1)
        if not k.isidentifier():
            bad.append(f"{k}: not a setting's name")
        elif v == "":
            nd.settings.pop(k, None)
        else:
            nd.settings[k] = value(v)
    return bad
