# Lesson 2: Sum a list

## The idea

In lesson 1 every thread worked alone. Adding up a list is different: every value has to meet every other one
in the end. A single thread walking the whole list would leave the rest of the card idle, so the threads have to
**work together**.

Two new tools:

- **Shared memory** (`__shared__`): a small piece of fast memory (up to 48 KB per block here) on the SM itself.
  The threads of one block all see the same shared memory; other blocks can't. It is about as fast as a
  register file and much faster than the card's main memory.
- **`__syncthreads()`**: a barrier. No thread of the block goes past it until every thread of the block has reached
  it. Without it, a thread could read `part[5]` before thread 5 has written it.

The usual shape is a **tree**: 256 values, then thread 0 adds part[0] + part[128], thread 1 adds part[1] + part[129]
... leaving 128 sums; then 64, 32, ... 1. That is 8 steps instead of 255, with the work spread out.

Each block then has one number. To put the blocks' numbers together, one thread of each block adds its sum into
`*out` with **`atomicAdd(out, value)`**: an addition the card makes sure two blocks can't do at the same moment
and lose one of them.

## Your task

Fill in the four steps in `sum`.

A warning that catches everyone: a `__syncthreads()` inside an `if` that only some threads enter will hang the
block (the others wait forever). If that happens, BitScryer stops the run after 30 seconds, and the card is fine.

## What to look at

- The answer isn't bit for bit the same as the CPU's: adding in a different order rounds differently. The check
  allows for that (a relative difference of 1e-4).
- In the SASS, find `BAR.SYNC` (your `__syncthreads`), `STS`/`LDS` (shared memory) and `RED` or `ATOM` (the atomic
  add).
- **Going further:** in the last five steps of the tree only one warp is still working, and a warp's threads are
  always in step, so they can trade values directly without shared memory: `__shfl_down_sync`. Our own kernels do
  exactly this (`warp_sum` in code/core/cuda/step.cu). Another trick: launch fewer blocks (`//! grid 112`) and have
  each thread first add up many values in a loop. Both make it faster; the timing will tell you by how much.
