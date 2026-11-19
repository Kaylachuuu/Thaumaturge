"""Lesson 5: unpacking 4-bit weights (Q4_K, repacked into separate arrays for the card)."""
import numpy as np

from thaumaturge.bitscryer.spec import Part, blocks_for

TITLE = "Unpack 4-bit weights"
SOURCE = "kernel.cu"
ROWS, CHUNKS = 4096, 120        # 120 chunks of 32: a row of 3,840 values, as wide as a 12B model's


def data(rng):
    n = ROWS * CHUNKS
    d = rng.uniform(2e-4, 2e-3, ROWS * CHUNKS // 8).astype(np.float16)
    dmin = rng.uniform(1e-4, 1e-3, ROWS * CHUNKS // 8).astype(np.float16)
    dd = d.view(np.uint16).astype(np.uint32) | (dmin.view(np.uint16).astype(np.uint32) << 16)
    sm = (rng.integers(0, 64, n) | (rng.integers(0, 64, n) << 8)).astype(np.uint16)
    return {"qs": rng.integers(0, 256, n * 16, dtype=np.uint8), "sm": sm, "dd": dd,
            "out": np.zeros(n * 32, np.float32), "n_rows": np.int32(ROWS), "n_chunk": np.int32(CHUNKS),
            "chunks": n}


def expect(d):
    """qmatvec.cu's q4_chunk, step by step in fp32: each multiply and subtract rounded on its own."""
    n = d["chunks"]
    qs = d["qs"].reshape(n, 16)
    s = d["sm"].astype(np.uint32)
    block = np.arange(n) // 8                      # dd: one per 8 chunks (256 values), rows run on (CHUNKS % 8 == 0)
    dd = d["dd"][block]
    dval = (dd & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    dmin = (dd >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    scale = (dval * (s & 0xFF).astype(np.float32)).astype(np.float32)
    mn = (dmin * (s >> 8).astype(np.float32)).astype(np.float32)
    lo = (qs & 15).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    v = np.empty((n, 32), np.float32)
    v[:, :16] = (scale[:, None] * lo).astype(np.float32) - mn[:, None]
    v[:, 16:] = (scale[:, None] * hi).astype(np.float32) - mn[:, None]
    return {"out": v.ravel()}


PARTS = [Part(kernel="unpack_q4k", args=["qs", "sm", "dd", "out", "n_rows", "n_chunk"], outputs=["out"],
              grid=blocks_for("chunks"), items=lambda d: d["chunks"], rtol=1e-6, atol=3e-7,
              bytes_moved=lambda d: d["chunks"] * (16 + 2 + 0.5 + 128), flops=lambda d: d["chunks"] * 64, goal=0.7)]
