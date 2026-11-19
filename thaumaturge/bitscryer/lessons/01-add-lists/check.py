"""Lesson 1: c = a + b, one thread for each value."""
import numpy as np

from thaumaturge.bitscryer.spec import Part, blocks_for, count

TITLE = "Add two lists"
SOURCE = "kernel.cu"
N = (1 << 24) + 77          # not a multiple of any block size, on purpose: the last block has threads to spare


def data(rng):
    return {"a": rng.standard_normal(N, dtype=np.float32), "b": rng.standard_normal(N, dtype=np.float32),
            "c": np.zeros(N, np.float32), "n": np.int32(N)}


def expect(d):
    return {"c": d["a"] + d["b"]}


PARTS = [Part(kernel="add", args=["a", "b", "c", "n"], outputs=["c"], grid=blocks_for("n"), items=count("n"),
              bytes_moved=lambda d: 3 * 4 * int(d["n"]), flops=lambda d: int(d["n"]))]
