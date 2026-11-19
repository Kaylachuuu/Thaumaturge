"""Lesson 7: a matrix multiply, plainly, then in tiles through shared memory."""
import math

import numpy as np

from thaumaturge.bitscryer.spec import Part

TITLE = "Matrix multiply, plain then tiled"
SOURCE = "kernel.cu"
M = N = K = 1024


def data(rng):
    return {"A": rng.standard_normal((M, K), dtype=np.float32), "B": rng.standard_normal((K, N), dtype=np.float32),
            "C": np.zeros((M, N), np.float32), "M": np.int32(M), "N": np.int32(N), "K": np.int32(K)}


def expect(d):
    return {"C": (d["A"].astype(np.float64) @ d["B"].astype(np.float64))}


def grid(d, block):
    return (math.ceil(int(d["N"]) / block), math.ceil(int(d["M"]) / block), 1)


def flops(d):
    return 2.0 * int(d["M"]) * int(d["N"]) * int(d["K"])


def bytes_moved(d):
    return 4.0 * (int(d["M"]) * int(d["K"]) + int(d["K"]) * int(d["N"]) + int(d["M"]) * int(d["N"]))


PARTS = [Part(kernel=k, args=["A", "B", "C", "M", "N", "K"], outputs=["C"], grid=grid, block=16, block_y=0,
              rtol=1e-4, atol=1e-3, flops=flops, bytes_moved=bytes_moved, items=lambda d: int(d["M"]) * int(d["N"]),
              goal=g, note=note)
         for k, g, note in (("matmul_plain", 0.0, "each thread one value of C, straight from memory"),
                            ("matmul_tiled", 0.25, "tiles of A and B shared by the block"))]
