#!/usr/bin/env python3
"""Checks BitScryer (thaumaturge/bitscryer/) without a card.

    python tests/test_bitscryer.py

1. Settings, and which cards it may use: a "never" card (by UUID, by its unit, or by size when the unit can't be
   read) never; a "busy" card only while it is free; no card while another program has it open.
2. Top speeds worked out from what a card says of itself (V100, T1000, 2080).
3. Reading the tools' output: settings lines, nvcc and ptxas errors, ptxas -v, PTX line marks, nvdisasm's SASS.
4. Every lesson's test data and CPU answer, and that an answer a card would give (summed in another order, or with
   a fused multiply-add) passes its check while a wrong one doesn't.
5. The picture of blocks and warps, and every screen drawn as text with a made-up run.
6. BitScryer's own editor: typing, Enter's indent, selection, undo and redo, cut, copy and paste, find, go to a
   line, Tab, saving; and the screen's keys (F-keys, Esc then a letter) reaching BitScryer, not the text.
7. If ptxas (and nvcc) are here: every answer and every starter file builds, and the SASS lines map back to the file.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["BITSCRYER_SETTINGS"] = os.devnull            # this machine's settings never change the tests
import numpy as np  # noqa: E402

from thaumaturge.bitscryer import build, cards, lessons, picture, settings, spec  # noqa: E402
from thaumaturge.bitscryer.runner import compare  # noqa: E402


def check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(what)
    print(f"ok  {what}")


KEPT = "GPU-aaaa1111-1111-2222-3333-444444444444"     # a card kept for other work: never used
SPARE = "GPU-bbbb2222-6e18-56e4-9d14-000000000002"    # a card lent out at times
T1000 = "GPU-cccc3333-0ebd-ca2f-0bb4-000000000003"
SMI = (f"0, {KEPT}, Tesla V100-PCIE-32GB, 32768, 20000, 7.0\n"
       f"1, {SPARE}, Tesla V100-PCIE-16GB, 16384, 9000, 7.0\n"
       f"2, {T1000}, NVIDIA T1000 8GB, 8192, 2, 7.5\n")
APPS = f"{KEPT}, 1234, /usr/bin/python\n{SPARE}, 2345, /usr/bin/python\n"
RULES = {
    "never": [{"card": "GPU-aaaa1111", "why": "the model server's card"},
              {"unit": "model.service", "else_over_gib": 30, "why": "the model server's card"}],
    "busy": [{"card": SPARE, "card_of_unit": "standby.service", "lock": "/run/lend.lock",
              "why": "lent to a training run right now"},
             {"card": SPARE, "card_of_unit": "standby.service", "unit_active": "painter.service",
              "why": "the painter is on it"},
             {"card": SPARE, "card_of_unit": "standby.service", "unit_active": "standby.service",
              "why": "the standby server is on it"}],
}


def world(units: dict | None = None, **state) -> "cards.World":
    """The rules above, with the units' cards (unreadable if left out), what's running, and which locks are held."""
    w = cards.World(never=RULES["never"], busy=RULES["busy"])
    w.unit_cards = dict(units or {})
    w.active = {u: state.get(u.split(".")[0], False) for u in ("painter.service", "standby.service")}
    w.locks = {"/run/lend.lock": state.get("lent", False)}
    return w


def test_settings() -> None:
    with tempfile.TemporaryDirectory() as d:
        machine, mine = Path(d) / "machine.json", Path(d) / "mine.json"
        import json
        machine.write_text(json.dumps({"cuda": "/opt/cuda", "cpus": "16-23", "never": RULES["never"]}))
        mine.write_text(json.dumps({"cpus": "0-3", "never": [{"card": "GPU-aaaa"}]}))
        s, problems = settings.load_with_problems([machine, mine])
        check(not problems and s["cuda"] == "/opt/cuda" and s["cpus"] == "0-3", "settings read; the user's win")
        check(len(s["never"]) == 3, "... but a rule list is added to, never replaced")
        mine.write_text("{not json")
        s, problems = settings.load_with_problems([machine, mine])
        check(problems and "not JSON" in problems[0] and len(s["never"]) == 2, "a broken file said, the rest kept")
        check(settings.load([Path(d) / "none.json"]) == {"never": [], "busy": []}, "no file: no rules")
        check(cards.gentle(["x"], {"cpus": "16-23"}) == ["taskset", "-c", "16-23", "nice", "-n", "10", "x"] and
              cards.gentle(["x"], {}) == ["nice", "-n", "10", "x"], "builds and runs kept to the chosen cores")
        # World.read asks systemd only about the units the rules name
        asked = []

        def run(argv):
            asked.append(argv)
            if argv[:2] == ["systemctl", "show"] and argv[2] == "model.service":
                return f"PATH=/bin CUDA_VISIBLE_DEVICES={KEPT}\n"
            if argv[:2] == ["systemctl", "is-active"]:
                return "active\n" if argv[2] == "painter.service" else "inactive\n"
            return ""
        w = cards.World.read(RULES, run)
        check(w.unit_cards == {"model.service": KEPT, "standby.service": ""} and w.active["painter.service"]
              and not w.active["standby.service"], "the rules' units read")
        check(cards.World.read({}, run).never == [] and len(asked) == 4, "no rules: nothing asked")


def test_cards() -> None:
    cs = cards.parse_cards(SMI)
    check([c.index for c in cs] == [0, 1, 2] and cs[2].cc == (7, 5) and cs[2].sm == "sm_75", "nvidia-smi read")
    check(cs[0].tensor_cores and cs[1].tensor_cores and not cs[2].tensor_cores, "the T1000 has no tensor cores")
    apps = cards.parse_apps(APPS)
    check(apps[SPARE] == ["python (2345)"], "programs on each card read")
    kept, spare, t1000 = cs
    w = world({"model.service": KEPT, "standby.service": SPARE})
    check(cards.never(kept, w) and "never used" in cards.why_not(kept, w), "the kept card refused")
    check(cards.never(kept, world()), "... by its UUID even if its unit can't be read")
    other_big = cards.Card(5, "GPU-ffffffff-0000", "A100 40GB", 40960, 0, (8, 0))
    check(cards.why_not(other_big, world()) != "" and
          cards.why_not(other_big, world({"model.service": KEPT})) == "",
          "a card of 30 GB or more refused only while the unit can't be read")
    check(cards.why_not(spare, w) == "" and cards.why_not(t1000, w) == "", "free cards allowed")
    for kw, word in (({"lent": True}, "training"), ({"painter": True}, "painter"), ({"standby": True}, "standby")):
        check(word in cards.why_not(spare, world({"model.service": KEPT}, **kw)),
              f"the lent card refused: {word}")
        check(word in cards.why_not(spare, world({"model.service": KEPT, "standby.service": SPARE}, **kw)),
              f"... also when found by its unit ({word})")
        check(cards.why_not(t1000, world({"model.service": KEPT}, **kw)) == "", f"... but not the T1000 ({word})")
    check(cards.why_not(kept, cards.World()) == "" and cards.why_not(t1000, cards.World()) == "",
          "no rules: free cards allowed")
    t1000.others = ["python (777)"]
    check("in use" in cards.why_not(t1000, w), "a card another program has open refused")
    check(cards.visible_uuid(spare) == SPARE, "a run sees only its own card, by its whole UUID")
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "lend.lock"
        check(not cards.lock_held(lock), "no lock file: not held")
        lock.write_text("")
        check(not cards.lock_held(lock), "a lock nobody holds: not held")
        p = subprocess.Popen([sys.executable, "-c", "import fcntl,sys,time; f=open(sys.argv[1],'a+'); "
                              "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); time.sleep(30)", str(lock)],
                             stdout=subprocess.PIPE, text=True)
        p.stdout.readline()
        check(cards.lock_held(lock), "a lock another program holds: held")
        p.kill()
        p.wait()


def test_peaks() -> None:
    bw = cards.peak_bandwidth(877_000, 4096)
    check(abs(bw / 1e9 - 898) < 2, f"V100 memory {bw / 1e9:.0f} GB/s")
    check(abs(cards.peak_fp32(80, 1_530_000, (7, 0)) / 1e12 - 15.7) < 0.1, "V100 fp32 15.7 TFLOPS")
    check(abs(cards.peak_tensor(80, 1_530_000, (7, 0), "Tesla V100-PCIE-16GB") / 1e12 - 125) < 1, "V100 tensor 125")
    check(abs(cards.peak_bandwidth(5_001_000, 128) / 1e9 - 160) < 1, "T1000 memory 160 GB/s")
    check(cards.peak_tensor(14, 1_395_000, (7, 5), "NVIDIA T1000 8GB") == 0, "T1000: no tensor cores")
    check(abs(cards.peak_bandwidth(7_000_000, 256) / 1e9 - 448) < 1, "2080 memory 448 GB/s")
    check(abs(cards.peak_tensor(46, 1_710_000, (7, 5), "NVIDIA GeForce RTX 2080") / 1e12 - 40.3) < 0.5,
          "2080 tensor (fp32 adds) half rate, 40 TFLOPS")


NVDISASM = """\t.headerflags\t@"EF_CUDA_SM75 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM75)"
//--------------------- .text.add                 --------------------------
\t.section\t.text.add,"ax",@progbits
\t.sectioninfo\t@"SHI_REGISTERS=12"
\t.align\t128
        .global         add
        .type           add,@function
add:
.text.add:
\t//## File "/home/k/bitscryer/lessons/01-add-lists/kernel.cu", line 1
        /*0000*/                   MOV R1, c[0x0][0x28] ;
\t//## File "/home/k/bitscryer/lessons/01-add-lists/kernel.cu", line 4
        /*0010*/                   S2R R6, SR_CTAID.X ;
        /*0040*/                   ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT ;
.L_x_1:
        /*0050*/               @P0 EXIT ;
\t//## File "/home/k/bitscryer/lessons/01-add-lists/kernel.cu", line 5
        /*0090*/                   LDG.E.SYS R4, [R4] ;
        /*00c0*/                   FADD R9, R4, R3 ;
.L_x_2:
        /*00e0*/                   EXIT ;
.L_x_0:
        /*00f0*/                   BRA `(.L_x_0);
        /*0100*/                   NOP;
        /*0110*/                   NOP;
.L_x_3:
"""


def test_reading() -> None:
    s = build.settings("//! block 128\n  //! block.softmax 512\n// not a setting\n//!grid 56\n")
    check(s == {"block": "128", "block.softmax": "512", "grid": "56"}, "settings lines read")
    check(build.setting_int(s, "block", "softmax", 1) == 512 and build.setting_int(s, "block", "add", 1) == 128
          and build.setting_int({}, "block", "add", 7) == 7, "a kernel's own setting first, then the file's")
    errs = build.parse_errors('/home/k/x/kernel.cu(4): error: expected a ";"\n'
                              '/usr/include/foo.h(9): error: something\n1 error detected in the compilation\n',
                              "kernel.cu")
    check(errs[0] == (4, 'error: expected a ";"') and errs[1][0] == 0, "nvcc's errors: her lines, and the rest")
    errs = build.parse_errors("ptxas /home/k/kernel.ptx, line 31; error   : Unknown symbol 'q'\n"
                              "ptxas fatal   : Ptx assembly aborted due to errors\n", "kernel.ptx")
    check(errs[0] == (31, "error   : Unknown symbol 'q'") and errs[1][0] == 0, "ptxas's errors on her PTX")
    u = build.parse_usage("ptxas info    : Compiling entry function 'add' for 'sm_75'\n"
                          "ptxas info    : Function properties for add\n"
                          "    0 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads\n"
                          "ptxas info    : Used 12 registers, used 1 barriers, 1024 bytes smem, 380 bytes cmem[0]\n")
    check(u == {"add": {"registers": 12, "shared": 1024, "stack": 0, "spill_stores": 8, "spill_loads": 8}},
          "ptxas -v: registers, shared, spills")
    ptx = build.parse_ptx('//\n// Generated by NVIDIA NVVM Compiler\n.version 8.7\n.file 1 "/k/kernel.cu"\n'
                          '.visible .entry add(\n.loc 1 4 5\nmov.u32 %r1, %ctaid.x;\n.loc 2 9 1\nadd.f32 %f1, %f2, '
                          '%f3;\n', "kernel.cu")
    check(ptx[0].text == ".version 8.7" and [ln.src for ln in ptx if ln.kind == "op"] == [4, 0],
          "PTX from CUDA: each instruction with its line of her file (another file's lines: 0)")
    check(build.parse_ptx("a\nb\n")[1].src == 2, "hand-written PTX: each line is its own")
    sass = build.parse_sass(NVDISASM, "kernel.cu")
    check([ln.text.split("  ", 1)[1] for ln in sass if ln.kind == "op"] ==
          ["MOV R1, c[0x0][0x28]", "S2R R6, SR_CTAID.X", "ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT", "@P0 EXIT",
           "LDG.E.SYS R4, [R4]", "FADD R9, R4, R3", "EXIT"], "SASS: the instructions, without the padding after")
    check([ln.src for ln in sass if ln.kind == "op"] == [1, 4, 4, 4, 5, 5, 5], "SASS: each with its line")
    check(sass[0].kind == "head" and not any(ln.kind == "label" for ln in sass),
          "labels nothing branches to left out")


def lesson_checks() -> dict[str, spec.Check]:
    return {w.key: spec.load(w.check) for w in lessons.lessons()}


def test_lessons() -> None:
    ws = lessons.lessons()
    check(len(ws) == 8 and [w.key[:2] for w in ws] == [f"0{i}" for i in range(1, 9)], "eight lessons, in order")
    check(ws[3].source.suffix == ".ptx" and ws[7].needs == ("tensor_cores",), "lesson 4 is PTX, 8 needs tensor cores")
    for w in ws:
        check((w.lesson / f"starter{w.source.suffix}").exists() and w.text() and w.hints(),
              f"{w.key}: starter, lesson text, hints")
        check((lessons.HERE / "answers" / f"{w.key}{w.source.suffix}").exists(), f"{w.key}: an answer for --selftest")
    chks = lesson_checks()
    rng = np.random.default_rng(0)

    def run(key, gpu_like, wrong=None):
        c = chks[key]
        d = c.data(np.random.default_rng(1234))
        want = c.expect(d)
        got = gpu_like(d)
        for p in c.parts:
            for o in p.outputs:
                r = compare(got[p.kernel][o], want[o], p.rtol, p.atol)
                check(r["right"], f"{key} {p.kernel}: an answer like a card's passes (max diff {r['max_diff']:.2g})")
                if wrong is not None:
                    bad = wrong(got[p.kernel][o].copy())
                    check(not compare(bad, want[o], p.rtol, p.atol)["right"], f"{key} {p.kernel}: a wrong one doesn't")
        return d, want

    flip = lambda a: (a.ravel().__setitem__(len(a.ravel()) - 1, a.ravel()[-1] + 1.0), a)[1]  # noqa: E731
    run("01-add-lists", lambda d: {"add": {"c": d["a"] + d["b"]}}, flip)
    run("04-add-in-ptx", lambda d: {"add": {"c": d["a"] + d["b"]}}, flip)

    def tree_sum(d):
        x = d["x"]
        parts = x.reshape(-1, 256)
        while parts.shape[1] > 1:                      # in float32, halving each step as the kernel does
            h = parts.shape[1] // 2
            parts = (parts[:, :h] + parts[:, h:]).astype(np.float32)
        total = np.float32(0)
        for v in rng.permutation(parts[:, 0]):         # blocks finish in any order
            total = np.float32(total + v)
        return {"sum": {"out": np.array([total], np.float32)}}
    run("02-sum-a-list", tree_sum, lambda a: a * 1.01)
    run("03-copy-fast", lambda d: {"copy_floats": {"dst": d["src"].copy()}, "copy4": {"dst": d["src"].copy()}},
        flip)

    def fused(d):                                      # d * scale * q - dmin * min, fused: rounded once
        n = d["chunks"]
        qs = d["qs"].reshape(n, 16)
        s = d["sm"].astype(np.uint32)
        dd = d["dd"][np.arange(n) // 8]
        dv = (dd & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        dm = (dd >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
        sc = (dv * (s & 0xFF).astype(np.float32)).astype(np.float32).astype(np.float64)
        mn = (dm * (s >> 8).astype(np.float32)).astype(np.float32).astype(np.float64)
        v = np.empty((n, 32), np.float64)
        v[:, :16] = sc[:, None] * (qs & 15) - mn[:, None]
        v[:, 16:] = sc[:, None] * (qs >> 4) - mn[:, None]
        return {"unpack_q4k": {"out": v.astype(np.float32).ravel()}}
    d, want = run("05-unpack-weights", fused, lambda a: a * 1.001 + 1e-5)
    got = fused(d)["unpack_q4k"]["out"]
    r = compare(got, want["out"], 0, 0)
    check(r["bad"] == 0, "05: a fused multiply-add gives the same bits here (the products need at most 21 bits: "
          "the lesson's puzzle)")

    def norm_softmax(d):
        x = d["x"].astype(np.float32)
        ss = np.zeros(x.shape[0], np.float32)
        for t in range(256):                           # each thread's part, then the parts added
            ss = (ss + (x[:, t::256] ** 2).sum(axis=1, dtype=np.float32)).astype(np.float32)
        r = (1 / np.sqrt(ss / np.float32(x.shape[1]) + np.float32(1e-6))).astype(np.float32)
        y = (x * r[:, None] * d["w"].astype(np.float32)).astype(np.float16)
        s = d["s"]
        e = np.exp(s - s.max(axis=1, keepdims=True)).astype(np.float32)
        p = (e / e.sum(axis=1, keepdims=True, dtype=np.float32)).astype(np.float32)
        return {"rms_norm": {"y": y}, "softmax": {"p": p}}
    run("06-norm-and-softmax", norm_softmax, lambda a: a * 1.01)

    def mm(d):
        c = (d["A"] @ d["B"]).astype(np.float32)
        return {"matmul_plain": {"C": c}, "matmul_tiled": {"C": c}}
    run("07-matmul", mm, flip)

    def wm(d):
        c = (d["A"].astype(np.float32) @ d["B"].astype(np.float32)).astype(np.float32)
        return {"mm_wmma": {"C": c}}
    run("08-tensor-cores", wm, flip)
    for key, c in chks.items():
        d = c.data(np.random.default_rng(0))
        for p in c.parts:
            g = p.grid(d, p.block)
            check(len(g) == 3 and all(int(v) >= 1 for v in g), f"{key} {p.kernel}: grid {g} for block {p.block}")
            for a in p.args:
                check(a in d, f"{key} {p.kernel}: argument {a} in the data")
    lesson8 = chks["08-tensor-cores"]
    d = lesson8.data(np.random.default_rng(0))
    check(lesson8.parts[0].grid(d, 128) == (1024, 1, 1), "lesson 8: 4,096 tiles, four warps a block, 1,024 blocks")


def test_projects() -> None:
    with tempfile.TemporaryDirectory() as t:
        w = lessons.new_project(Path(t) / "mine")
        check(w.source.name == "kernel.cu" and w.check and w.lesson is None, "a new project: kernel and check.py")
        c = spec.load(w.check)
        d = c.data(np.random.default_rng(0))
        check(compare(d["k"] * d["x"], c.expect(d)["y"], 1e-6, 0)["right"], "the template's check works")
        try:
            lessons.new_project(Path(t) / "mine")
            check(False, "a project never written over")
        except FileExistsError:
            check(True, "a project never written over")
        (Path(t) / "loose").mkdir()
        (Path(t) / "loose" / "thing.ptx").write_text("// mine\n")
        w = lessons.project(Path(t) / "loose")
        check(w.source.name == "thing.ptx" and w.check is None, "a folder with just a kernel: built, not run")
        p = lessons.Progress(Path(t) / "p.json")
        p.mark_passed("01-add-lists")
        p.see_hint("02-sum-a-list", 2)
        q = lessons.Progress(Path(t) / "p.json")
        check(q.passed("01-add-lists") and q.hints_seen("02-sum-a-list") == 2, "progress kept")
        ws = lessons.lessons(Path(t))
        lessons.open_lesson(ws[0])
        ws[0].source.write_text("mine")
        lessons.open_lesson(ws[0])
        check(ws[0].source.read_text() == "mine", "her lesson file never written over")


INFO = {"name": "NVIDIA T1000 8GB", "sms": 14, "clock_khz": 1395000, "max_threads_sm": 1024, "regs_sm": 65536,
        "shared_sm": 65536, "max_blocks_sm": 16, "max_threads_block": 1024, "peak_bw": 160e9, "peak_fp32": 2.5e12,
        "peak_tensor": 0}


def test_picture() -> None:
    part = {"kernel": "add", "grid": [65537, 1, 1], "block": [256, 1, 1], "items": (1 << 24) + 77, "registers": 12,
            "shared": 0, "shared_dyn": 0, "blocks_per_sm": 4}
    text = "\n".join(t for t, _ in picture.picture(part, INFO))
    check("65,537 blocks × 256 threads" in text and "8 warps" in text, "the launch, in blocks and warps")
    check("The last 179 have nothing to do" in text and "the last block has 77 working" in text,
          "the idle threads at the end")
    check("4 blocks per SM, 56 at a time" in text and "1,171 waves" in text, "how it lands on the card")
    lim = picture.limits(INFO, 256, 128, 0)
    check(lim["registers"] == 2 and lim["threads"] == 4, "128 registers a thread: two blocks of 256 per SM")
    text = "\n".join(t for t, _ in picture.picture({**part, "block": [100, 1, 1], "grid": [10, 1, 1], "items": 5000},
                                                   INFO))
    check("only 4 threads" in text and "each thread must take more than one" in text, "odd blocks and grid-stride")


def test_screen() -> None:
    from thaumaturge.common.screen import TextCanvas
    from thaumaturge.bitscryer import screen
    with tempfile.TemporaryDirectory() as t:
        os.environ["BITSCRYER_HOME"] = t
        cards.read_cards = lambda run=None: cards.parse_cards(SMI)            # no nvidia-smi here
        cards.World.read = classmethod(lambda cls, settings=None, run=None: world(
            {"model.service": KEPT, "standby.service": SPARE}, standby=True))
        app = screen.App()
        check(app.card and app.card.uuid == T1000, "the T1000 chosen: the lent card is busy, the kept one never")
        app2 = screen.App(card_pick="0")
        check(app2.card is None or not app2.card.uuid.startswith("GPU-aaaa1111"), "asking for the kept card never gets it")
        app.info = INFO
        app.built = build.Built(ok=True, sass=[build.Line("add:", 0, "head"), build.Line("0000  S2R R0", 12)],
                                ptx=[build.Line("mov.u32 %r1, %ctaid.x;", 12)])
        app.cursor = 12
        app.result = {"card": INFO, "parts": [
            {"kernel": "add", "status": "right", "grid": [65537, 1, 1], "block": [256, 1, 1], "items": 16777293,
             "registers": 12, "shared": 0, "shared_dyn": 0, "local": 0, "blocks_per_sm": 4,
             "outputs": {"c": {"right": True, "n": 16777293, "exact": 16777293, "bad": 0}},
             "ms_median": 1.42, "gbs": 141.7, "of_bw": 0.886, "gflops": 11.8, "of_flops": 0.005, "goal": 0}]}
        app.status = "right"
        for ov in (None, "lesson", "hints", "warps", "cards", "lessons", "help", "log"):
            app.overlay = ov
            app.progress.see_hint(app.work.key, 1)
            for size in ((100, 30), (200, 50)):
                cv = TextCanvas(size[1], size[0])
                app.draw(cv)
                text = cv.text()
                check(len(text) > 100, f"screen {ov or 'main'} at {size[0]}x{size[1]}")
            if ov is None:
                check("✓ right" in text and "89% of 160 GB/s" in text and "▶0000  S2R R0" in text,
                      "results, speed, and the cursor's SASS lit")
            if ov == "cards":
                check("never used" in text and "standby" in text and "free to use" in text,
                      "the cards screen says why each card can or can't be used")
                check("7 of the 8 lessons run" in text, "... and how many lessons this card runs")
        app.result = {"error": "the kernel ran: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered"}
        app.overlay = None
        cv = TextCanvas(30, 120)
        app.draw(cv)
        check("outside its arrays" in cv.text(), "a crash explained plainly")


def test_editor() -> None:
    import curses
    from thaumaturge.common.editor import Editor, colour_spans
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "kernel.cu"
        f.write_text("int a;\n")
        ed = Editor(f)

        def keys(*ks):
            for k in ks:
                ed.key(k)
        check(ed.lines == ["int a;"] and not ed.dirty, "a file opened")
        keys(curses.KEY_END, "\n")
        for c in "void f() {":
            ed.key(c)
        keys("\n")
        check(ed.lines[-1] == "    " and ed.col == 4, "Enter after { indents")
        for c in "x = 1;":
            ed.key(c)
        keys("\n", "}")
        check(ed.lines[-2:] == ["    x = 1;", "}"], "a closing brace lines up with its line")
        check(ed.dirty, "changed, not saved")
        keys("\x13")
        check(f.read_text() == "int a;\nvoid f() {\n    x = 1;\n}\n" and not ed.dirty, "Ctrl-S saves")
        keys("\x1a")
        check(ed.lines[-1] == "    ", "Ctrl-Z undoes the last step (the brace)")
        keys("\x1a")
        check(ed.lines[-1] == "    x = 1;", "and the one before (Enter)")
        keys("\x19", "\x19")
        check(ed.lines[-1] == "}", "Ctrl-Y redoes")
        ed.goto(2, 4)
        keys(curses.KEY_SRIGHT, curses.KEY_SRIGHT, curses.KEY_SRIGHT, curses.KEY_SRIGHT, curses.KEY_SRIGHT)
        check(ed.selected_text() == "x = 1", "Shift-arrows select")
        keys("\x03")
        keys("y")
        check(ed.lines[2] == "    y;", "typing replaces the selection")
        keys(curses.KEY_HOME, "\x16")
        check(ed.lines[2] == "    x = 1y;", "Ctrl-V pastes what was copied")
        ed.goto(0)
        keys("\x03", "\x16")
        check(ed.lines[:2] == ["int a;", "int a;"], "Ctrl-C with nothing selected copies the line; paste puts it above")
        keys("\x0b")
        check(ed.lines[0] == "int a;" and ed.lines[1] == "void f() {", "Ctrl-K deletes the line")
        keys("\x04")
        check(ed.lines[1:3] == ["void f() {", "void f() {"] and ed.row == 2, "Ctrl-D duplicates it")
        keys("\x06", "x", " ", "=", "\n")
        check(ed.row == 3 and ed.selected_text() == "x =" and ed.prompt is None, "Ctrl-F finds and selects")
        keys("\x07", "1", "\n")
        check(ed.row == 0, "Ctrl-G goes to a line")
        keys("\t")
        check(ed.lines[0] == "    int a;", "Tab indents")
        keys(curses.KEY_BTAB)
        check(ed.lines[0] == "int a;", "Shift-Tab takes it back")
        keys(curses.KEY_END, 127, 127)
        check(ed.lines[0] == "int ", "Backspace")
        ed.goto(0, 0)
        keys(curses.KEY_DOWN, curses.KEY_HOME, curses.KEY_LEFT, curses.KEY_DC)
        check(ed.lines[0] == "int void f() {", "Delete at a line's end joins the next")
    spans = colour_spans("    int i = blockIdx.x * 4; // the index", ptx=False)
    names = {c for _, _, c in spans}
    check({"attn", "mem", "cyan", "muted"} <= names, "CUDA coloured: keywords, threadIdx and friends, numbers, comments")
    check(any(c == "mem" for _, _, c in colour_spans("mov.u32 %r1, %ctaid.x;", ptx=True)), "PTX registers coloured")


def test_screen_keys() -> None:
    import curses
    from thaumaturge.bitscryer import screen
    with tempfile.TemporaryDirectory() as t:
        os.environ["BITSCRYER_HOME"] = t
        app = screen.App()
        before = list(app.editor.lines)
        app.key(curses.KEY_F1)
        check(app.overlay == "lesson", "F1 opens the lesson")
        app.key("j")
        check(app.editor.lines == before, "keys in a page don't reach the text")
        app.key(curses.KEY_F1)
        check(app.overlay is None, "F1 again closes it")
        app.key("l", alt=True)
        check(app.overlay == "lesson", "Esc then l does the same")
        app.key("\x1b")
        check(app.overlay is None, "Esc closes it")
        app.key(curses.KEY_F2)
        check(app.overlay == "hints" and app.progress.hints_seen(app.work.key) == 1, "F2: the first hint")
        app.key(curses.KEY_F2)
        check(app.progress.hints_seen(app.work.key) == 2, "F2 again: the next")
        app.key(curses.KEY_F3)
        check(app.overlay == "warps", "another command switches page")
        app.key("\x1b")
        app.key("x")
        check(app.editor.dirty and "x" in "".join(app.editor.lines), "letters go to the text")
        check(app.key("\x11") is False and not app.editor.dirty, "Ctrl-Q quits, saving first")


def test_build() -> None:
    if not build.cuda_home() and not shutil.which("ptxas"):
        print("--  no ptxas here: building skipped (install CUDA, or set BITSCRYER_CUDA)")
        return
    have_nvcc = (build.cuda_home() / "bin" / "nvcc").exists() if build.cuda_home() else bool(shutil.which("nvcc"))
    srcs = sorted(p for p in (lessons.HERE / "answers").iterdir() if p.suffix in (".cu", ".ptx")) + sorted(lessons.LESSONS.glob("*/starter.*")) + \
        [lessons.TEMPLATE / "kernel.cu"]
    for f in srcs:
        if f.suffix == ".cu" and not have_nvcc:
            continue
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / ("kernel" + f.suffix)
            shutil.copyfile(f, src)
            b = build.build(src, "sm_75")
            check(b.ok and b.sass and b.usage, f"{f.parent.name}/{f.name} builds for sm_75 "
                                               f"({', '.join(f'{k} {u['registers']} regs' for k, u in b.usage.items())})")
            if f.suffix == ".cu" and "answers" in str(f):
                lines = len(src.read_text().splitlines())
                check(any(0 < ln.src <= lines for ln in b.sass), f"{f.name}: SASS lines map back to the file")


if __name__ == "__main__":
    test_settings()
    test_cards()
    test_peaks()
    test_reading()
    test_lessons()
    test_projects()
    test_picture()
    test_screen()
    test_editor()
    test_screen_keys()
    test_build()
    print("all ok")
