"""How a lesson (or a project of your own) says what to run and how to check it: its check.py.

A check.py is ordinary Python with numpy. It names:

    TITLE   = "Add two lists"
    SOURCE  = "kernel.cu"            # or "kernel.ptx"
    NEEDS   = []                     # ["tensor_cores"] if the card must have them
    PARTS   = [Part(...), ...]       # one per kernel to run
    def data(rng):    -> dict        # the test data: numpy arrays (sent to the card) and numpy scalars
    def expect(d):    -> dict        # what the outputs should be, worked out on the CPU

A Part says which kernel to run, its arguments by name (in the kernel's order), which of them are outputs, how to
size the grid from the block, and how much work one run does (so the timing can be held against the card's top
speed). Outputs are put back to their starting values from data() before the run that is checked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


def count(n_key: str, per_thread: int = 1) -> Callable[[dict], int]:
    """Threads with work: one for each of d[n_key] items, per_thread items each."""
    return lambda d: math.ceil(int(d[n_key]) / per_thread)


def blocks_for(n_key: str, per_thread: int = 1) -> Callable[[dict, int], tuple]:
    """A grid with one thread for each of d[n_key] items (per_thread items each), the last block partly idle."""
    def grid(d: dict, block: int) -> tuple:
        return (max(1, math.ceil(int(d[n_key]) / (block * per_thread))), 1, 1)
    return grid


def fixed(x: int, y: int = 1, z: int = 1) -> Callable[[dict, int], tuple]:
    return lambda d, block: (x, y, z)


@dataclass
class Part:
    kernel: str
    args: list[str]
    outputs: list[str]
    grid: Callable[[dict, int], tuple] = field(default=fixed(1))
    block: int = 256                                  # used unless your file says `//! block N`
    shared: int = 0                                   # dynamic shared memory in bytes
    bytes_moved: Callable[[dict], float] | None = None   # bytes read and written by one run, at best
    flops: Callable[[dict], float] | None = None         # arithmetic in one run
    rtol: float = 0.0
    atol: float = 0.0
    goal: float = 0.0                                 # a fraction of the top speed to aim for (shown, not required)
    tensor: bool = False                              # its flops are held against the tensor cores' top speed
    may_set_grid: bool = False                        # `//! grid N` may replace the grid
    block_y: int = 1                                  # 2D blocks: (block, block_y) threads; 0: square
    items: Callable[[dict], int] | None = None         # how many threads have work (for the picture), if known
    note: str = ""


@dataclass
class Check:
    title: str
    source: str
    parts: list[Part]
    data: Callable
    expect: Callable
    needs: list[str] = field(default_factory=list)
    repeat_s: float = 0.3                             # how long to spend timing each part


def load(path) -> Check:
    """A check.py read into a Check."""
    import runpy
    g = runpy.run_path(str(path), run_name="bitscryer_check")
    return Check(title=g.get("TITLE", path.parent.name if hasattr(path, "parent") else "?"),
                 source=g.get("SOURCE", "kernel.cu"), parts=list(g["PARTS"]), data=g["data"], expect=g["expect"],
                 needs=list(g.get("NEEDS", [])), repeat_s=float(g.get("REPEAT_S", 0.3)))
