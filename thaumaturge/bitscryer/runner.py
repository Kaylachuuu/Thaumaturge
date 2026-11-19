"""Running your kernel on the card and checking it: a program of its own, started fresh for every run.

    CUDA_VISIBLE_DEVICES=GPU-... python3 -m thaumaturge.bitscryer.runner run  CHECK.py KERNEL.cubin SETTINGS.json
    CUDA_VISIBLE_DEVICES=GPU-... python3 -m thaumaturge.bitscryer.runner info

It prints one line of JSON and exits. Being its own program is what keeps the screen safe: a kernel that reads
memory it shouldn't kills only this program, and one that never ends is stopped from outside (the screen's time
limit), which frees the card. It talks to the card's driver directly (libcuda, through ctypes), with no PyTorch, so
there is nothing between your code and the card but the driver.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from thaumaturge.bitscryer import cards  # noqa: E402
from thaumaturge.bitscryer.build import setting_int  # noqa: E402
from thaumaturge.bitscryer.spec import load  # noqa: E402

ATTR = {"sms": 16, "clock_khz": 13, "mem_clock_khz": 36, "bus_bits": 37, "major": 75, "minor": 76,
        "max_threads_block": 1, "max_threads_sm": 39, "shared_block": 8, "shared_sm": 81, "regs_sm": 82,
        "l2": 38, "max_blocks_sm": 106}
F_REGS, F_SHARED, F_LOCAL = 4, 1, 3


class CudaError(Exception):
    pass


class Driver:
    def __init__(self):
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise CudaError(f"can't load the card's driver (libcuda.so.1): {e}")
        self.ck(self.lib.cuInit(0), "cuInit")
        n = ctypes.c_int()
        self.ck(self.lib.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
        if n.value != 1:
            raise CudaError(f"expected to see exactly one card, saw {n.value}: CUDA_VISIBLE_DEVICES isn't set to one")
        self.dev = ctypes.c_int()
        self.ck(self.lib.cuDeviceGet(ctypes.byref(self.dev), 0), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        self.ck(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.dev), "cuDevicePrimaryCtxRetain")
        self.ck(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    def ck(self, r: int, what: str) -> None:
        if r:
            name, text = ctypes.c_char_p(), ctypes.c_char_p()
            self.lib.cuGetErrorName(r, ctypes.byref(name))
            self.lib.cuGetErrorString(r, ctypes.byref(text))
            n = name.value.decode() if name.value else str(r)
            t = text.value.decode() if text.value else ""
            raise CudaError(f"{what}: {n}: {t}")

    def attr(self, which: int) -> int:
        v = ctypes.c_int()
        self.ck(self.lib.cuDeviceGetAttribute(ctypes.byref(v), which, self.dev), "cuDeviceGetAttribute")
        return v.value

    def info(self) -> dict:
        name = ctypes.create_string_buffer(256)
        self.ck(self.lib.cuDeviceGetName(name, 256, self.dev), "cuDeviceGetName")
        a = {k: self.attr(v) for k, v in ATTR.items()}
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.ck(self.lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
        cc = (a["major"], a["minor"])
        nm = name.value.decode()
        return {"name": nm, "uuid": os.environ.get("CUDA_VISIBLE_DEVICES", ""), **a, "cc": list(cc),
                "free": free.value, "total": total.value,
                "peak_bw": cards.peak_bandwidth(a["mem_clock_khz"], a["bus_bits"]),
                "peak_fp32": cards.peak_fp32(a["sms"], a["clock_khz"], cc),
                "peak_tensor": cards.peak_tensor(a["sms"], a["clock_khz"], cc, nm)}

    def alloc(self, nbytes: int) -> int:
        p = ctypes.c_uint64()
        self.ck(self.lib.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(max(1, nbytes))), "cuMemAlloc")
        return p.value

    def put(self, dst: int, a: np.ndarray) -> None:
        a = np.ascontiguousarray(a)
        self.ck(self.lib.cuMemcpyHtoD_v2(ctypes.c_uint64(dst), a.ctypes.data_as(ctypes.c_void_p),
                                         ctypes.c_size_t(a.nbytes)), "copying to the card")

    def get(self, src: int, like: np.ndarray) -> np.ndarray:
        out = np.empty_like(like)
        self.ck(self.lib.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(src),
                                         ctypes.c_size_t(out.nbytes)), "copying back from the card")
        return out

    def module(self, image: bytes):
        m = ctypes.c_void_p()
        self.ck(self.lib.cuModuleLoadData(ctypes.byref(m), ctypes.c_char_p(image)), "loading the kernel")
        return m

    def function(self, module, name: str):
        f = ctypes.c_void_p()
        r = self.lib.cuModuleGetFunction(ctypes.byref(f), module, name.encode())
        return f if r == 0 else None

    def fattr(self, f, which: int) -> int:
        v = ctypes.c_int()
        self.ck(self.lib.cuFuncGetAttribute(ctypes.byref(v), which, f), "cuFuncGetAttribute")
        return v.value

    def occupancy(self, f, block: int, shared: int) -> int:
        v = ctypes.c_int()
        r = self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(v), f, block, ctypes.c_size_t(shared))
        return v.value if r == 0 else 0

    def launch(self, f, grid, block, shared: int, params) -> None:
        self.ck(self.lib.cuLaunchKernel(f, *[ctypes.c_uint(g) for g in grid], *[ctypes.c_uint(b) for b in block],
                                        ctypes.c_uint(shared), None, params, None), "launching the kernel")

    def sync(self, what: str = "the kernel ran") -> None:
        self.ck(self.lib.cuCtxSynchronize(), what)

    def event(self):
        e = ctypes.c_void_p()
        self.ck(self.lib.cuEventCreate(ctypes.byref(e), 0), "cuEventCreate")
        return e

    def record(self, e) -> None:
        self.ck(self.lib.cuEventRecord(e, None), "cuEventRecord")

    def elapsed(self, a, b) -> float:
        ms = ctypes.c_float()
        self.ck(self.lib.cuEventSynchronize(b), "the timed runs")
        self.ck(self.lib.cuEventElapsedTime(ctypes.byref(ms), a, b), "cuEventElapsedTime")
        return ms.value


SCALARS = {np.dtype(np.int32): ctypes.c_int32, np.dtype(np.int64): ctypes.c_int64,
           np.dtype(np.uint32): ctypes.c_uint32, np.dtype(np.uint64): ctypes.c_uint64,
           np.dtype(np.float32): ctypes.c_float, np.dtype(np.float64): ctypes.c_double}


def params_for(args: list[str], d: dict, ptrs: dict[str, int]):
    """The kernel's argument list as the driver wants it: an array of pointers, one to each argument's value."""
    vals = []
    for name in args:
        v = d[name]
        if isinstance(v, np.ndarray):
            vals.append(ctypes.c_uint64(ptrs[name]))
        else:
            v = np.asarray(v)
            vals.append(SCALARS[v.dtype](v.item()))
    arr = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
    return arr, vals


def compare(got: np.ndarray, want: np.ndarray, rtol: float, atol: float) -> dict:
    """How close got is to want: right if every value is within atol + rtol * |want|. The first few that aren't."""
    same_kind = np.asarray(want).dtype == got.dtype     # bit for bit means something only then
    g = got.astype(np.float64).ravel()
    w = np.asarray(want).astype(np.float64).ravel()
    if g.shape != w.shape:
        return {"right": False, "detail": f"{g.size} values came back, {w.size} expected", "bad": 0, "first": []}
    diff = np.abs(g - w)
    bad_mask = ~(diff <= atol + rtol * np.abs(w))          # NaN counts as wrong
    bad = int(bad_mask.sum())
    first = [{"at": int(i), "got": float(g[i]), "want": float(w[i])} for i in np.flatnonzero(bad_mask)[:6]]
    finite = np.isfinite(diff)
    exact = int((g == w).sum()) if same_kind else None
    return {"right": bad == 0, "bad": bad, "n": int(g.size), "first": first, "exact": exact,
            "max_diff": float(diff[finite].max()) if finite.any() else float("nan")}


def run(check_path: str, cubin_path: str, settings_path: str) -> dict:
    chk = load(Path(check_path))
    s = json.loads(Path(settings_path).read_text()) if settings_path else {}
    drv = Driver()
    info = drv.info()
    out = {"card": info, "parts": []}
    if "tensor_cores" in chk.needs and not info["peak_tensor"]:
        out["error"] = "this card has no tensor cores, so this lesson can't run on it"
        return out
    mod = drv.module(Path(cubin_path).read_bytes())
    rng = np.random.default_rng(1234)
    t0 = time.time()
    d = chk.data(rng)
    want = chk.expect(d)
    out["data_s"] = round(time.time() - t0, 2)
    arrays = {k: v for k, v in d.items() if isinstance(v, np.ndarray)}
    need = sum(v.nbytes for v in arrays.values())
    if need > info["free"] * 0.8:
        out["error"] = f"the test data needs {need / 2**20:.0f} MiB and the card has {info['free'] / 2**20:.0f} free"
        return out
    ptrs = {k: drv.alloc(v.nbytes) for k, v in arrays.items()}
    for k, v in arrays.items():
        drv.put(ptrs[k], v)
    for part in chk.parts:
        r = {"kernel": part.kernel, "note": part.note}
        out["parts"].append(r)
        f = drv.function(mod, part.kernel)
        if f is None:
            r["status"] = "missing"
            r["detail"] = f"no kernel called {part.kernel} in the file (is it `extern \"C\" __global__`?)"
            continue
        block = setting_int(s, "block", part.kernel, part.block)
        grid = part.grid(d, block)
        if part.may_set_grid and setting_int(s, "grid", part.kernel, 0) > 0:
            grid = (setting_int(s, "grid", part.kernel, 0), 1, 1)
        by = part.block_y or block                 # 0: a square block
        bdim = (block, by, 1)
        if part.items:
            r["items"] = int(part.items(d))
        r.update({"grid": list(grid), "block": list(bdim), "shared_dyn": part.shared,
                  "registers": drv.fattr(f, F_REGS), "shared": drv.fattr(f, F_SHARED), "local": drv.fattr(f, F_LOCAL),
                  "blocks_per_sm": drv.occupancy(f, block * by, part.shared)})
        if block * by > info["max_threads_block"] or block < 1:
            r["status"] = "error"
            r["detail"] = f"a block of {block * by} threads: this card allows 1 to {info['max_threads_block']}"
            continue
        for name in part.outputs:
            drv.put(ptrs[name], arrays[name])
        params, keep = params_for(part.args, d, ptrs)
        drv.launch(f, grid, bdim, part.shared, params)
        drv.sync("the kernel ran")
        right = True
        r["outputs"] = {}
        for name in part.outputs:
            c = compare(drv.get(ptrs[name], arrays[name]), want[name], part.rtol, part.atol)
            r["outputs"][name] = c
            right = right and c["right"]
        r["status"] = "right" if right else "wrong"
        # timing: a couple of warm-up runs, then as many as fit in repeat_s, each timed on the card itself
        for _ in range(2):
            drv.launch(f, grid, bdim, part.shared, params)
        drv.sync("the warm-up runs")
        e0, e1 = drv.event(), drv.event()
        drv.record(e0)
        drv.launch(f, grid, bdim, part.shared, params)
        drv.record(e1)
        first = drv.elapsed(e0, e1)
        n = int(max(3, min(500, chk.repeat_s * 1000 / max(first, 1e-3))))
        times = []
        for _ in range(n):
            drv.record(e0)
            drv.launch(f, grid, bdim, part.shared, params)
            drv.record(e1)
            times.append(drv.elapsed(e0, e1))
        times.sort()
        best, med = times[0], times[len(times) // 2]
        r.update({"ms_best": best, "ms_median": med, "runs": n})
        if part.bytes_moved:
            b = float(part.bytes_moved(d))
            r["bytes"] = b
            r["gbs"] = b / (med * 1e-3) / 1e9
            r["of_bw"] = (b / (med * 1e-3)) / info["peak_bw"] if info["peak_bw"] else None
        if part.flops:
            fl = float(part.flops(d))
            peak = info["peak_tensor"] if part.tensor else info["peak_fp32"]
            r["flops"] = fl
            r["gflops"] = fl / (med * 1e-3) / 1e9
            r["of_flops"] = (fl / (med * 1e-3)) / peak if peak else None
        r["goal"] = part.goal
        del keep
    return out


def main(argv: list[str]) -> int:
    try:
        if argv[:1] == ["info"]:
            res = {"card": Driver().info()}
        elif argv[:1] == ["run"] and len(argv) >= 3:
            res = run(argv[1], argv[2], argv[3] if len(argv) > 3 else "")
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except CudaError as e:
        res = {"error": str(e)}
    except Exception as e:                       # your check.py, or ours: say where
        res = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
    print(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
