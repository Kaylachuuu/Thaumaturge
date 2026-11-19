"""Lesson 6: two of the small steps between a language model's matrix products: RMSNorm (as Gemma does it) and softmax."""
import numpy as np

from thaumaturge.bitscryer.spec import Part

TITLE = "Norm and softmax"
SOURCE = "kernel.cu"
ROWS, WIDTH = 1024, 3840        # 1,024 tokens of a 12B model's width
SROWS, SWIDTH = 1024, 4096      # 1,024 rows of attention scores over 4,096 places
EPS = 1e-6


def data(rng):
    return {"x": (rng.standard_normal((ROWS, WIDTH)) * 3).astype(np.float16),
            "w": rng.uniform(0.5, 1.5, WIDTH).astype(np.float16),
            "y": np.zeros((ROWS, WIDTH), np.float16), "n": np.int32(WIDTH), "eps": np.float32(EPS),
            "rows": ROWS,
            "s": (rng.standard_normal((SROWS, SWIDTH)) * 4).astype(np.float32),
            "p": np.zeros((SROWS, SWIDTH), np.float32), "sn": np.int32(SWIDTH), "srows": SROWS}


def expect(d):
    x = d["x"].astype(np.float32)
    r = 1.0 / np.sqrt((x * x).sum(axis=1, keepdims=True) / np.float32(WIDTH) + np.float32(EPS))
    y = (x * r * d["w"].astype(np.float32)).astype(np.float16)
    s = d["s"].astype(np.float64)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return {"y": y, "p": (e / e.sum(axis=1, keepdims=True)).astype(np.float32)}


PARTS = [Part(kernel="rms_norm", args=["x", "w", "y", "n", "eps"], outputs=["y"],
              grid=lambda d, block: (d["rows"], 1, 1), rtol=2e-3, atol=1e-4,
              bytes_moved=lambda d: d["rows"] * int(d["n"]) * 4 + int(d["n"]) * 2,
              flops=lambda d: d["rows"] * int(d["n"]) * 4, note="one block per row"),
         Part(kernel="softmax", args=["s", "p", "sn"], outputs=["p"], grid=lambda d, block: (d["srows"], 1, 1),
              rtol=1e-4, atol=1e-9, bytes_moved=lambda d: d["srows"] * int(d["sn"]) * 8,
              flops=lambda d: d["srows"] * int(d["sn"]) * 5, note="one block per row")]
