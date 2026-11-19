"""Lesson 8: the tensor cores, through wmma: a warp multiplies 16 x 16 tiles of fp16 in one instruction."""
import math

import numpy as np

from thaumaturge.bitscryer.spec import Part

TITLE = "Tensor cores"
SOURCE = "kernel.cu"
NEEDS = ["tensor_cores"]
M = N = K = 1024


def data(rng):
    return {"A": (rng.standard_normal((M, K)) * 0.5).astype(np.float16),
            "B": (rng.standard_normal((K, N)) * 0.5).astype(np.float16),
            "C": np.zeros((M, N), np.float32), "M": np.int32(M), "N": np.int32(N), "K": np.int32(K)}


def expect(d):
    return {"C": d["A"].astype(np.float64) @ d["B"].astype(np.float64)}


def grid(d, block):
    tiles = (int(d["M"]) // 16) * (int(d["N"]) // 16)
    return (math.ceil(tiles / max(1, block // 32)), 1, 1)


PARTS = [Part(kernel="mm_wmma", args=["A", "B", "C", "M", "N", "K"], outputs=["C"], grid=grid, block=128,
              rtol=1e-3, atol=1e-3, tensor=True, flops=lambda d: 2.0 * int(d["M"]) * int(d["N"]) * int(d["K"]),
              bytes_moved=lambda d: 2.0 * (int(d["M"]) + int(d["N"])) * int(d["K"]) + 4.0 * int(d["M"]) * int(d["N"]),
              goal=0.15, note="one warp per 16 x 16 tile of C")]
