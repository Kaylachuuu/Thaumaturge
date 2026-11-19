"""Which cards BitScryer may run on, and how fast each one could go at best.

With no settings, a card is refused only while another program has it open. A machine whose cards do other work
says more in its settings file (see settings.py and examples/bitscryer.json):

- **never**: cards never used, however they are asked for: by UUID (or its start), or the card a systemd unit is
  pinned to (its CUDA_VISIBLE_DEVICES). If that unit can't be read, `else_over_gib` refuses every card that big
  instead, so a card kept for something else stays safe even then.
- **busy**: cards that are lent out at times: refused while a lock file is held (a shared flock can't be taken) or
  while a systemd unit is running.

A run sees only the card it was given (CUDA_VISIBLE_DEVICES set to that card's UUID), and these checks are made again
before every run, not just when the card is chosen.
"""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from thaumaturge.bitscryer.settings import load as load_settings

# Cards whose chip has no tensor cores although its compute capability would say it has (Turing's small chips).
NO_TENSOR = re.compile(r"\b(T400|T550|T600|T1000|T1200|GTX 16\d\d|MX\d{3})\b", re.I)
# Plain fp32 cores per multiprocessor, by compute capability.
CORES_PER_SM = {(6, 0): 64, (6, 1): 128, (7, 0): 64, (7, 2): 64, (7, 5): 64, (8, 0): 64, (8, 6): 128, (8, 7): 128,
                (8, 9): 128, (9, 0): 128}


@dataclass
class Card:
    index: int
    uuid: str
    name: str
    mem_total: float                       # MiB
    mem_used: float
    cc: tuple[int, int]                    # compute capability, (7, 5) for sm_75
    others: list[str] = field(default_factory=list)   # other programs on the card, "name (pid)"

    @property
    def sm(self) -> str:
        return f"sm_{self.cc[0]}{self.cc[1]}"

    @property
    def tensor_cores(self) -> bool:
        return self.cc >= (7, 0) and not NO_TENSOR.search(self.name)

    def short(self) -> str:
        name = self.name.replace("NVIDIA ", "").replace("Tesla ", "").replace("GeForce ", "")
        return f"GPU{self.index} {name}" + ("" if re.search(r"\d+\s*GB", name) else f" {self.mem_total / 1024:.0f}GB")


def _run(argv: list[str], timeout: float = 10) -> str:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout if r.returncode == 0 else ""


def parse_cards(text: str) -> list[Card]:
    """nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,compute_cap (csv, no header, no units)."""
    out = []
    for line in text.splitlines():
        p = [s.strip() for s in line.split(",")]
        if len(p) != 6:
            continue
        try:
            major, _, minor = p[5].partition(".")
            out.append(Card(int(p[0]), p[1], p[2], float(p[3]), float(p[4]), (int(major), int(minor or 0))))
        except ValueError:
            continue
    return out


def parse_apps(text: str) -> dict[str, list[str]]:
    """nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name: each card's programs."""
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        p = [s.strip() for s in line.split(",")]
        if len(p) >= 3 and p[0].startswith("GPU-"):
            out.setdefault(p[0], []).append(f"{os.path.basename(p[2]) or '?'} ({p[1]})")
    return out


def read_cards(run: Callable[[list[str]], str] = _run) -> list[Card]:
    cards = parse_cards(run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,compute_cap",
                             "--format=csv,noheader,nounits"]))
    apps = parse_apps(run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                           "--format=csv,noheader"]))
    for c in cards:
        c.others = [a for a in apps.get(c.uuid, []) if not a.endswith(f"({os.getpid()})")]
    return cards


def unit_card(unit: str, run: Callable[[list[str]], str] = _run) -> str:
    """The card a systemd unit is pinned to (its CUDA_VISIBLE_DEVICES), or ""."""
    for item in run(["systemctl", "show", unit, "-p", "Environment", "--value"]).split():
        if item.startswith("CUDA_VISIBLE_DEVICES="):
            return item.split("=", 1)[1]
    return ""


def unit_active(unit: str, run: Callable[[list[str]], str] = _run) -> bool:
    return run(["systemctl", "is-active", unit]).strip() == "active"


def lock_held(lock: str | Path) -> bool:
    """True while another program holds this lock file (a shared flock can't be taken)."""
    try:
        f = open(lock, "r")
    except OSError:
        return False                       # no lock file: never taken since the machine started
    try:
        fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    except OSError:
        return False
    finally:
        f.close()


@dataclass
class World:
    """The rules (from the settings) and what they need to know, read once; each part replaceable in the tests."""
    never: list[dict] = field(default_factory=list)
    busy: list[dict] = field(default_factory=list)
    unit_cards: dict[str, str] = field(default_factory=dict)    # unit -> its card ("" if it can't be read)
    active: dict[str, bool] = field(default_factory=dict)       # unit -> running
    locks: dict[str, bool] = field(default_factory=dict)        # lock file -> held

    @classmethod
    def read(cls, settings: dict | None = None, run: Callable[[list[str]], str] = _run) -> "World":
        s = load_settings() if settings is None else settings
        w = cls(never=list(s.get("never") or []), busy=list(s.get("busy") or []))
        for r in w.never + w.busy:
            for key in ("unit", "card_of_unit"):
                if r.get(key) and r[key] not in w.unit_cards:
                    w.unit_cards[r[key]] = unit_card(r[key], run)
            if r.get("unit_active") and r["unit_active"] not in w.active:
                w.active[r["unit_active"]] = unit_active(r["unit_active"], run)
            if r.get("lock") and r["lock"] not in w.locks:
                w.locks[r["lock"]] = lock_held(r["lock"])
        return w


def never_why(card: Card, world: World) -> str:
    """Why this card is never to be used ("" if it may be, at some time)."""
    for r in world.never:
        why = r.get("why") or "kept for something else"
        if r.get("card") and card.uuid.startswith(r["card"]):
            return f"{why}: never used"
        if r.get("unit"):
            pinned = world.unit_cards.get(r["unit"], "")
            if pinned and card.uuid == pinned:
                return f"{why}: never used"
            if not pinned and r.get("else_over_gib") and card.mem_total >= float(r["else_over_gib"]) * 1024:
                return (f"can't tell which card {r['unit']} uses, so no card of {r['else_over_gib']} GB or more "
                        f"is used")
    return ""


def never(card: Card, world: World) -> bool:
    return bool(never_why(card, world))


def why_not(card: Card, world: World) -> str:
    """Why a run may not use this card right now, in a line; "" if it may."""
    why = never_why(card, world)
    if why:
        return why
    for r in world.busy:
        target = (world.unit_cards.get(r["card_of_unit"], "") if r.get("card_of_unit") else "") or r.get("card", "")
        if not target or not card.uuid.startswith(target):
            continue
        if r.get("lock") and world.locks.get(r["lock"]):
            return r.get("why") or f"lent out right now ({r['lock']} is held)"
        if r.get("unit_active") and world.active.get(r["unit_active"]):
            return r.get("why") or f"{r['unit_active']} is using it"
    if card.others:
        return "in use by " + ", ".join(card.others[:3])
    return ""


def visible_uuid(card: Card) -> str:
    """The value CUDA_VISIBLE_DEVICES gets for a run: the card's whole UUID, so no other card can be seen."""
    return card.uuid


def experiment_cpus(settings: dict | None = None) -> str:
    """The cores builds and runs are kept to: BITSCRYER_CPUS, else the settings' "cpus", else none (any)."""
    named = os.environ.get("BITSCRYER_CPUS")
    if named is None:
        named = str((load_settings() if settings is None else settings).get("cpus") or "")
    return "" if named.strip().lower() in ("", "all") else named.strip()


def gentle(argv: list[str], settings: dict | None = None) -> list[str]:
    """A command run on the chosen cores (if any) at a lower priority, so it never takes CPU from the machine's own
    work."""
    s = load_settings() if settings is None else settings
    cpus = experiment_cpus(s)
    pin = ["taskset", "-c", cpus] if cpus else []
    return [*pin, "nice", "-n", str(int(s.get("nice", 10))), *argv]


# -- top speeds, from what the card says of itself (the runner reads these with the driver)

def peak_bandwidth(mem_clock_khz: int, bus_bits: int) -> float:
    """Bytes per second the memory could move at best: two transfers a clock, bus_bits wide."""
    return 2.0 * mem_clock_khz * 1e3 * bus_bits / 8.0


def peak_fp32(sms: int, clock_khz: int, cc: tuple[int, int]) -> float:
    """Plain fp32 flops per second at best: every core one fused multiply-add (two flops) each clock."""
    return 2.0 * sms * CORES_PER_SM.get(tuple(cc), 64) * clock_khz * 1e3


def peak_tensor(sms: int, clock_khz: int, cc: tuple[int, int], name: str) -> float:
    """fp16 tensor-core flops per second at best, adding in fp32 (0 for a card without them). Volta and Turing do
    1,024 flops per multiprocessor each clock; GeForce Turing cards do half that when adding in fp32."""
    if cc < (7, 0) or NO_TENSOR.search(name):
        return 0.0
    rate = 1024.0
    if cc == (7, 5) and re.search(r"GeForce|RTX 20", name, re.I):
        rate /= 2
    return sms * clock_khz * 1e3 * rate
