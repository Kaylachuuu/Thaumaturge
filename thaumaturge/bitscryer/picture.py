"""A picture of how a launch splits into blocks and warps, and how they land on the card: the biggest change of
thinking coming from a CPU.

Lines are (text, colour name) pairs; the screen draws them, and the tests read them as plain text.
"""

from __future__ import annotations

import math


def _n(x: float) -> str:
    return f"{int(x):,}"


def limits(card: dict, block: int, registers: int, shared: int) -> dict[str, int]:
    """How many of these blocks one multiprocessor (SM) could hold at once, by each thing that runs out."""
    out = {}
    if block > 0:
        out["threads"] = card.get("max_threads_sm", 1024) // block
        warps = math.ceil(block / 32)
        if registers:
            # registers are handed out per warp, in steps of 256 (Volta and Turing)
            per_warp = math.ceil(registers * 32 / 256) * 256
            out["registers"] = card.get("regs_sm", 65536) // (per_warp * warps)
        if shared:
            out["shared memory"] = card.get("shared_sm", 65536) // shared
        out["blocks"] = card.get("max_blocks_sm", 16) or 16
    return out


def picture(part: dict, card: dict, width: int = 100) -> list[tuple[str, str]]:
    L: list[tuple[str, str]] = []
    grid = part.get("grid") or [1, 1, 1]
    bdim = part.get("block") or [1, 1, 1]
    blocks = grid[0] * grid[1] * grid[2]
    block = bdim[0] * bdim[1] * bdim[2]
    threads = blocks * block
    warps = math.ceil(block / 32)
    L.append((f"{part.get('kernel', '?')}:  a grid of {_n(blocks)} blocks × {_n(block)} threads = {_n(threads)} "
              f"threads", "text"))
    if grid[1] > 1 or grid[2] > 1 or bdim[1] > 1:
        L.append((f"  (grid {grid[0]}×{grid[1]}×{grid[2]}, block {bdim[0]}×{bdim[1]}×{bdim[2]}: blockIdx and "
                  f"threadIdx have .x, .y and .z)", "muted"))
    L.append(("", "text"))
    L.append((f"One block: {_n(block)} threads = {warps} warp{'s' if warps != 1 else ''} of 32", "attn"))
    show = min(warps, 8)
    for w in range(show):
        lo, hi = w * 32, min(block, w * 32 + 32) - 1
        full = hi - lo + 1
        L.append((f"  warp {w:<3} threads {lo:>4}–{hi:<4} {'█' * full}{'░' * (32 - full)}", "ff"))
    if warps > show:
        L.append((f"  ... {warps - show} more warps", "muted"))
    if block % 32:
        L.append((f"  The last warp has only {block % 32} threads: the card still runs it as 32, so "
                  f"{32 - block % 32} lanes do nothing. Blocks are best a multiple of 32.", "warn"))
    L.append(("  A warp's 32 threads run each instruction together, side by side. When an `if` sends some one way",
              "muted"))
    L.append(("  and some the other, the warp runs both ways, one after the other, each lane sitting out the one",
              "muted"))
    L.append(("  that isn't its own.", "muted"))
    items = part.get("items")
    if items is not None and threads:
        idle = threads - items
        L.append(("", "text"))
        if idle > 0:
            last = block - idle % block if idle % block else 0
            L.append((f"Threads with work: {_n(items)} of {_n(threads)}. The last {_n(idle)} have nothing to do "
                      f"(they stop at your `if`){f': the last block has {last} working' if last else ''}.", "warn"))
        elif idle < 0:
            L.append((f"Only {_n(threads)} threads for {_n(items)} items: each thread must take more than one "
                      f"(a loop that steps by the whole grid).", "attn"))
        else:
            L.append((f"Every thread has exactly one item: {_n(items)}.", "ok"))
    sms = card.get("sms") or 0
    if sms and block:
        L.append(("", "text"))
        lim = limits(card, block, part.get("registers", 0), (part.get("shared", 0) or 0) + (part.get("shared_dyn", 0) or 0))
        held = part.get("blocks_per_sm") or (min(lim.values()) if lim else 0)
        why = min(lim, key=lim.get) if lim else ""
        L.append((f"The card: {sms} multiprocessors (SMs). Each holds at most:", "attn"))
        for k, v in lim.items():
            mark = "  ◀ the limit" if k == why else ""
            L.append((f"  {v:>3} blocks, by {k}{mark}", "warn" if k == why else "text"))
        if held:
            at_once = held * sms
            waves = math.ceil(blocks / at_once)
            occ = held * warps / max(1, card.get("max_threads_sm", 1024) // 32)
            L.append((f"So {held} block{'s' if held != 1 else ''} per SM, {_n(at_once)} at a time on the card: "
                      f"{_n(waves)} wave{'s' if waves != 1 else ''}. Warps busy per SM: {held * warps} of "
                      f"{card.get('max_threads_sm', 1024) // 32} ({occ:.0%}).", "text"))
            rows = min(sms, 12)
            cols = max(1, min(held, (width - 14) // 2))
            L.append(("", "text"))
            L.append(("The first wave (each ▣ one of your blocks):", "muted"))
            placed = 0
            for s in range(rows):
                here = min(held, max(0, blocks - s * held)) if blocks < at_once else held
                placed += here
                L.append((f"  SM {s:>2}  " + "▣ " * min(here, cols) + ("…" if here > cols else "") +
                          ("· " * (held - here) if here < held and held <= cols else ""), "ff"))
            if sms > rows:
                L.append((f"  ... {sms - rows} more SMs, the same", "muted"))
            if waves > 1:
                tail = blocks - (waves - 1) * at_once
                L.append((f"The last wave has {_n(tail)} blocks, so {at_once - tail:,} places sit empty while it "
                          f"finishes.", "muted" if tail * 2 > at_once else "warn"))
    return L
