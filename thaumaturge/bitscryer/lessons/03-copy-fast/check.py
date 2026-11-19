"""Lesson 3: copying memory as fast as the card allows, first a float at a time, then four."""
import numpy as np

from thaumaturge.bitscryer.spec import Part, blocks_for, count

TITLE = "Copy at full speed"
SOURCE = "kernel.cu"
N = 1 << 26                 # 64 Mi floats: 256 MiB each way, far bigger than the card's caches


def data(rng):
    return {"src": rng.standard_normal(N, dtype=np.float32), "dst": np.zeros(N, np.float32),
            "n": np.int32(N), "n4": np.int32(N // 4)}


def expect(d):
    return {"dst": d["src"]}


PARTS = [Part(kernel="copy_floats", args=["src", "dst", "n"], outputs=["dst"], grid=blocks_for("n"),
              items=count("n"), may_set_grid=True, bytes_moved=lambda d: 8 * int(d["n"]), goal=0.75,
              note="one float per thread"),
         Part(kernel="copy4", args=["src", "dst", "n4"], outputs=["dst"], grid=blocks_for("n4"), items=count("n4"),
              may_set_grid=True, bytes_moved=lambda d: 8 * int(d["n"]), goal=0.80, note="a float4 per thread")]
