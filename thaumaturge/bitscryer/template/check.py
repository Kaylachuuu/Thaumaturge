"""How BitScryer runs and checks this project's kernel. Change it to suit yours; any lesson's check.py
(thaumaturge/bitscryer/lessons/*/check.py) is a worked example, and thaumaturge/bitscryer/spec.py says what every field means.

data(rng) makes the test data: numpy arrays go to the card (and outputs are put back to these values before the
checked run), numpy scalars are passed as they are (np.int32 is an int, np.float32 a float). expect(d) works out on
the CPU what each output should be. PARTS lists the kernels to run, with their arguments in order.
"""
import numpy as np

from thaumaturge.bitscryer.spec import Part, blocks_for, count

TITLE = "My kernel"
SOURCE = "kernel.cu"
NEEDS = []                    # ["tensor_cores"] if it uses them
N = 1 << 20


def data(rng):
    return {"x": rng.standard_normal(N, dtype=np.float32), "y": np.zeros(N, np.float32),
            "k": np.float32(2.5), "n": np.int32(N)}


def expect(d):
    return {"y": d["k"] * d["x"]}


PARTS = [Part(kernel="scale", args=["x", "y", "k", "n"], outputs=["y"], grid=blocks_for("n"), items=count("n"),
              rtol=1e-6, bytes_moved=lambda d: 8 * int(d["n"]), flops=lambda d: int(d["n"]))]
