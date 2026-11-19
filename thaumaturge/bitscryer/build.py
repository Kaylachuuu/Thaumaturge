"""Turning your file into something the card runs, and showing every step of the way.

    kernel.cu  --nvcc -ptx-->  kernel.ptx  --ptxas-->  kernel.cubin  --nvdisasm-->  SASS (the card's own instructions)
    kernel.ptx (written by hand)  --ptxas-->  kernel.cubin  --nvdisasm-->  SASS

Every step's output is kept in the lesson's `.build/` folder, so you can open them too. The PTX from CUDA carries
`.loc` lines (which line of your file each instruction came from), and nvdisasm passes them on, so the screen can show
which PTX and which SASS came from the line your cursor is on.

Settings for a run are comment lines in your own file, starting `//!`:

    //! block 256           threads in a block, for every kernel in the file
    //! block.softmax 512   ... for one kernel only
    //! grid 160            blocks in the grid (if the lesson lets you choose; most work it out from the block)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from thaumaturge.bitscryer.cards import gentle
from thaumaturge.bitscryer.settings import load as load_settings

SETTING = re.compile(r"^\s*//!\s*([A-Za-z_][\w.]*)\s+(\S+)")


def settings(text: str) -> dict[str, str]:
    """Your `//! key value` lines."""
    out = {}
    for line in text.splitlines():
        m = SETTING.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def setting_int(s: dict[str, str], key: str, kernel: str, default: int) -> int:
    v = s.get(f"{key}.{kernel}", s.get(key))
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def cuda_home() -> Path | None:
    """Where nvcc and friends are: BITSCRYER_CUDA, the settings' "cuda", CUDA_HOME, /usr/local/cuda, or wherever
    ptxas is on the PATH."""
    for c in (os.environ.get("BITSCRYER_CUDA"), load_settings().get("cuda"), os.environ.get("CUDA_HOME"),
              "/usr/local/cuda"):
        if c and (Path(c) / "bin" / "ptxas").exists():
            return Path(c)
    found = shutil.which("ptxas")
    return Path(found).resolve().parent.parent if found else None


def tool(name: str) -> str:
    home = cuda_home()
    return str(home / "bin" / name) if home and (home / "bin" / name).exists() else name


def host_compiler() -> list[str]:
    """nvcc runs the host's C++ compiler even for -ptx, and each CUDA accepts gcc only up to some version (12.8: 14),
    while a newer system's own may be newer: BITSCRYER_HOST_CXX or the settings' "host_cxx" if set, else a g++-14 or
    13 if one is installed, else the newer one allowed."""
    host = os.environ.get("BITSCRYER_HOST_CXX") or load_settings().get("host_cxx") or shutil.which("g++-14") \
        or shutil.which("g++-13")
    return ["-ccbin", host] if host else ["-allow-unsupported-compiler"]


@dataclass
class Line:
    """One line of PTX or SASS: its text, and the line of your file it came from (0: unknown)."""
    text: str
    src: int = 0
    kind: str = "op"           # "op" an instruction, "label", "head" a function's start, "dir" a directive


@dataclass
class Built:
    ok: bool = False
    stage: str = ""                                  # where it stopped: "nvcc", "ptxas", "" if it got through
    errors: list[tuple[int, str]] = field(default_factory=list)   # (line of your file or 0, message)
    log: str = ""
    ptx: list[Line] = field(default_factory=list)
    sass: list[Line] = field(default_factory=list)
    usage: dict[str, dict] = field(default_factory=dict)     # kernel -> registers, shared, stack, spills
    cubin: Path | None = None
    seconds: float = 0.0


ERR = re.compile(r"^(?P<file>[^\s:(][^:(]*?)(?:\((?P<l1>\d+)\)|:(?P<l2>\d+)(?::\d+)?)?:?\s*(?P<msg>(?:error|warning|"
                 r"fatal|ptxas fatal|ptxas error|ptxas warning).*)$", re.I)


def parse_errors(text: str, source_name: str) -> list[tuple[int, str]]:
    """nvcc's and ptxas's complaints as (line, message), lines of your own file only (0 for the rest)."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(.*?)\((\d+)\):\s*(.*)$", line)                 # nvcc: kernel.cu(12): error: ...
        if m and os.path.basename(m.group(1)) == source_name:
            out.append((int(m.group(2)), m.group(3)))
            continue
        m = re.match(r"^ptxas\s+(.*?),\s*line\s+(\d+);\s*(.*)$", line)  # ptxas kernel.ptx, line 12; error : ...
        if m:
            n = int(m.group(2)) if os.path.basename(m.group(1)) == source_name else 0
            out.append((n, m.group(3)))
            continue
        if re.search(r"error|fatal", line, re.I):
            out.append((0, line))
    return out


USAGE_FN = re.compile(r"Compiling entry function '([^']+)'|Function properties for (\S+)")
USAGE_REG = re.compile(r"Used (\d+) registers")
USAGE_SMEM = re.compile(r"(\d+) bytes smem")
USAGE_STACK = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")


def parse_usage(text: str) -> dict[str, dict]:
    """ptxas -v: each kernel's registers, shared memory, stack and spills."""
    out: dict[str, dict] = {}
    cur = None
    for line in text.splitlines():
        m = USAGE_FN.search(line)
        if m:
            cur = m.group(1) or m.group(2)
            out.setdefault(cur, {"registers": 0, "shared": 0, "stack": 0, "spill_stores": 0, "spill_loads": 0})
            continue
        if cur is None:
            continue
        m = USAGE_REG.search(line)
        if m:
            out[cur]["registers"] = int(m.group(1))
        m = USAGE_SMEM.search(line)
        if m:
            out[cur]["shared"] = int(m.group(1))
        m = USAGE_STACK.search(line)
        if m:
            out[cur]["stack"], out[cur]["spill_stores"], out[cur]["spill_loads"] = map(int, m.groups())
    return out


LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)")
FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"')


def parse_ptx(text: str, source_name: str | None = None) -> list[Line]:
    """PTX as lines. From CUDA, each instruction gets the line of your file its last `.loc` named, and the
    `.loc`/`.file` lines themselves are left out; written by hand, every line is your own line (src = its number)."""
    files: dict[str, str] = {}
    out: list[Line] = []
    cur = 0
    for n, raw in enumerate(text.splitlines(), 1):
        m = FILE.match(raw)
        if m:
            files[m.group(1)] = os.path.basename(m.group(2))
            continue
        m = LOC.match(raw)
        if m:
            cur = int(m.group(2)) if (source_name is None or files.get(m.group(1)) == source_name) else 0
            continue
        s = raw.rstrip().replace("\t", "    ")
        st = s.strip()
        kind = ("label" if re.match(r"^\$?[\w$]+:$", st) else "head" if st.startswith((".visible", ".entry",
                ".func", ".extern")) else "dir" if st.startswith(".") or st.startswith("//") or not st else "op")
        out.append(Line(s, cur if source_name is not None else n, kind))
    if source_name is not None:                    # nvcc's own heading ("Generated by NVIDIA NVVM Compiler" ...)
        while out and (out[0].text.strip().startswith("//") or not out[0].text.strip()):
            out.pop(0)
    return out


SASS_LINE = re.compile(r'//## File "([^"]+)", line (\d+)')
SASS_OP = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s*(.*?)\s*;?\s*(/\*.*\*/)?\s*$")


def parse_sass(text: str, source_name: str) -> list[Line]:
    """nvdisasm -c -g: the instructions, each with the line of your file it came from; the section bookkeeping left
    out. For hand-written PTX there are no line marks, so src stays 0."""
    out: list[Line] = []
    cur = 0
    for raw in text.splitlines():
        m = SASS_LINE.search(raw)
        if m:
            cur = int(m.group(2)) if os.path.basename(m.group(1)) == source_name else 0
            continue
        st = raw.strip()
        if not st or st.startswith(("//-", ".section", ".sectioninfo", ".align", ".global", ".type", ".size",
                                    ".other", ".target", ".elftype", ".text.", ".headerflags", ".nv.")):
            if st.startswith(".text.") and not st.endswith(":"):
                continue
            if not st.startswith(".text."):
                continue
        m = SASS_OP.match(raw)
        if m:
            out.append(Line(f"{m.group(1)}  {m.group(2)}", cur, "op"))
        elif st.endswith(":") and not st.startswith(".text."):
            kind = "label" if st.startswith(".L") else "head"
            out.append(Line(st, 0, kind))
    out = drop_padding(out)
    used = {m.group(1) for ln in out if ln.kind == "op" for m in [re.search(r"`\((\.L_x_\d+)\)", ln.text)] if m}
    return [ln for ln in out if ln.kind != "label" or ln.text[:-1] in used]   # labels nothing branches to, left out


def drop_padding(lines: list[Line]) -> list[Line]:
    """Each function's end as the card lays it out: a branch to itself after the last EXIT (a trap, never reached),
    then NOPs to fill out 128 bytes. Neither is your code, so they are left out."""
    out: list[Line] = []
    padding = False
    for i, ln in enumerate(lines):
        if ln.kind == "head":
            padding = False
        if not padding and ln.kind == "label" and i + 1 < len(lines) and \
                lines[i + 1].text.split("  ", 1)[-1].strip() == f"BRA `({ln.text[:-1]})":
            padding = True
        if padding and (ln.kind == "label" or "NOP" in ln.text or "BRA" in ln.text):
            continue
        padding = False if ln.kind == "op" else padding
        out.append(ln)
    return out


def _call(argv: list[str], cwd: Path, timeout: float = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(gentle(argv), cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 1, "", f"error: {argv[0]} took over {timeout:.0f} s")
    except OSError as e:
        return subprocess.CompletedProcess(argv, 1, "", f"error: couldn't start {argv[0]}: {e}")


def build(source: Path, sm: str, out_dir: Path | None = None) -> Built:
    """Your file (.cu or .ptx) built for one card (sm, such as "sm_75"), every step kept in .build/."""
    import time
    t0 = time.time()
    out_dir = out_dir or source.parent / ".build"
    out_dir.mkdir(parents=True, exist_ok=True)
    b = Built()
    name = source.name
    logs = []
    need = (["nvcc"] if source.suffix == ".cu" else []) + ["ptxas"]
    missing = [t for t in need if not (Path(tool(t)).exists() or shutil.which(tool(t)))]
    if missing:
        b.stage = missing[0]
        b.errors = [(0, f"{missing[0]} not found: install the CUDA toolkit, or say where it is with \"cuda\" in "
                        f"~/.config/thaumaturge/bitscryer.json (or BITSCRYER_CUDA)")]
        b.log = b.errors[0][1]
        b.seconds = time.time() - t0
        return b
    if source.suffix == ".cu":
        ptx = out_dir / "kernel.ptx"
        r = _call([tool("nvcc"), *host_compiler(), f"-arch={sm.replace('sm_', 'compute_')}", "-ptx", "-lineinfo",
                   "-O3", "-std=c++17", "-Wno-deprecated-gpu-targets", "-o", str(ptx), str(source)], source.parent)
        logs.append(r.stdout + r.stderr)
        if r.returncode != 0:
            b.stage, b.errors, b.log = "nvcc", parse_errors(r.stderr + r.stdout, name), "\n".join(logs)
            b.seconds = time.time() - t0
            return b
        from_cuda = True
    else:
        ptx = source
        from_cuda = False
    cubin = out_dir / "kernel.cubin"
    r = _call([tool("ptxas"), f"-arch={sm}", "-v", "-O3", "--generate-line-info", "-o", str(cubin), str(ptx)],
              source.parent)
    logs.append(r.stdout + r.stderr)
    b.usage = parse_usage(r.stdout + r.stderr)
    if r.returncode != 0:
        b.stage, b.errors, b.log = "ptxas", parse_errors(r.stderr + r.stdout, name if not from_cuda else "kernel.ptx"), \
            "\n".join(logs)
        if from_cuda:                     # ptxas complaining about nvcc's PTX: lines of the PTX, not yours
            b.errors = [(0, m) for _, m in b.errors]
        b.seconds = time.time() - t0
        return b
    try:
        b.ptx = parse_ptx(ptx.read_text(errors="replace"), name if from_cuda else None)
    except OSError:
        pass
    r = _call([tool("nvdisasm"), "-c", "-g", str(cubin)], source.parent)
    if r.returncode != 0:
        r2 = _call([tool("cuobjdump"), "-sass", str(cubin)], source.parent)
        text = r2.stdout if r2.returncode == 0 else ""
        logs.append(r.stderr)
    else:
        text = r.stdout
    (out_dir / "kernel.sass").write_text(text)
    b.sass = parse_sass(text, name if from_cuda else "")
    b.ok, b.cubin, b.log = True, cubin, "\n".join(logs)
    b.seconds = time.time() - t0
    return b
