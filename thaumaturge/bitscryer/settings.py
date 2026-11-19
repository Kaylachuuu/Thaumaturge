"""BitScryer's settings for this machine: which cards to leave alone, where CUDA is, which cores to use.

Read from /etc/thaumaturge/bitscryer.json (the machine's, set by whoever looks after it), then
~/.config/thaumaturge/bitscryer.json (yours). BITSCRYER_SETTINGS names one file to read instead of both. A rule list
("never", "busy") in your file is added to the machine's, never put in its place, so a card the machine keeps for
something else can't be freed from a user's file. Everything else in yours wins. Every key is optional; with no file
at all, BitScryer finds CUDA itself and refuses only a card another program has open. examples/bitscryer.json shows
them all:

    cuda        the CUDA folder (with bin/nvcc), if it isn't found by itself (BITSCRYER_CUDA wins)
    host_cxx    the C++ compiler nvcc should use, if the newest installed is too new for it (BITSCRYER_HOST_CXX)
    cpus        cores for builds and runs, as taskset takes them ("16-23"), so the machine's own work keeps the rest
    nice        how much lower their priority is (10 if not set)
    never       cards never used: [{"card": "GPU-1234abcd", "why": "..."},
                                   {"unit": "x.service", "else_over_gib": 30, "why": "..."}]
    busy        cards lent out at times: [{"card": "GPU-...", "lock": "/run/x.lock", "why": "..."},
                                          {"card_of_unit": "y.service", "unit_active": "y.service", "why": "..."}]
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MACHINE = Path("/etc/thaumaturge/bitscryer.json")
LISTS = ("never", "busy")


def user_file() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "thaumaturge" / "bitscryer.json"


def read_one(path: Path) -> tuple[dict, str]:
    """A settings file's contents, and what was wrong with it ("" if nothing)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, ""
    except OSError as e:
        return {}, f"{path}: can't be read ({e.strerror})"
    except ValueError as e:
        return {}, f"{path}: not JSON as written ({e})"
    if not isinstance(data, dict):
        return {}, f"{path}: should be a JSON object ({{...}})"
    for k in LISTS:
        if k in data and not (isinstance(data[k], list) and all(isinstance(r, dict) for r in data[k])):
            return {}, f"{path}: {k} should be a list of {{...}} rules"
    return data, ""


def load_with_problems(paths: list[Path] | None = None) -> tuple[dict, list[str]]:
    if paths is None:
        named = os.environ.get("BITSCRYER_SETTINGS")
        paths = [Path(named)] if named else [MACHINE, user_file()]
    out: dict = {k: [] for k in LISTS}
    problems = []
    for p in paths:
        data, problem = read_one(p)
        if problem:
            problems.append(problem)
        for k, v in data.items():
            if k in LISTS:
                out[k].extend(v)
            else:
                out[k] = v
    return out, problems


def load(paths: list[Path] | None = None) -> dict:
    return load_with_problems(paths)[0]
