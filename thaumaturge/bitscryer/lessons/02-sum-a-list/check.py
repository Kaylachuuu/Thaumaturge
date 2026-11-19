"""Lesson 2: the sum of a list, with the threads of a block working together in shared memory."""
import numpy as np

from thaumaturge.bitscryer.spec import Part, blocks_for, count

TITLE = "Sum a list"
SOURCE = "kernel.cu"
N = 1 << 22


def data(rng):
    return {"x": rng.random(N, dtype=np.float32), "out": np.zeros(1, np.float32), "n": np.int32(N)}


def expect(d):
    return {"out": np.array([d["x"].astype(np.float64).sum()], np.float64)}


PARTS = [Part(kernel="sum", args=["x", "out", "n"], outputs=["out"], grid=blocks_for("n"), items=count("n"),
              may_set_grid=True, rtol=1e-4, bytes_moved=lambda d: 4 * int(d["n"]), flops=lambda d: int(d["n"]))]
