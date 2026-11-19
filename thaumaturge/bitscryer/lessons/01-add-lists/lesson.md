# Lesson 1: Add two lists

## The idea

On a CPU you'd write a loop: `for (i = 0; i < n; i++) c[i] = a[i] + b[i];`. One core walks through all
sixteen million values, one after another.

A GPU has no loop here. Instead it starts **one thread for every value**, all running the same function (a
*kernel*), and each thread does just one addition. The T1000 has 896 cores, so hundreds of additions happen at
the same moment.

The threads come in groups:

- A **block** is a group of threads you choose the size of (here 256). Threads in the same block can work together
  (lesson 2).
- The **grid** is all the blocks of one launch. For 16,777,293 values and 256 threads a block, BitScryer
  launches 65,537 blocks.
- Inside each block, the card runs threads 32 at a time, side by side, in **warps**. A warp is the real unit
  the hardware schedules: one instruction, 32 lanes. (Press F3 after a run to see it drawn.)

Each thread finds out where it is from built-in variables: `threadIdx.x` (its number in the block), `blockIdx.x`
(its block's number) and `blockDim.x` (threads per block). From those, it works out **which value is its own**.

## Your task

Fill in `add` so that every value of `c` is `a + b`.

Two things to get right:

1. Which index is this thread's? Think of the blocks laid end to end.
2. The number of values isn't a multiple of 256 (on purpose), so the last block has threads with nothing to do.
   They mustn't touch memory past the end of the lists.

## What to look at once it's right

- **The SASS pane.** Move the cursor (↑↓) to your lines and see the card's own instructions light up. Find the
  two loads (`LDG`), the add (`FADD`) and the store (`STG`). Find `S2R`: it reads the thread's and block's numbers
  out of special registers.
- **The speed.** Adding is nearly free; nearly all the time goes to reading `a` and `b` and writing `c`, so the
  toolbox holds the time against the card's memory speed (GB/s), not its arithmetic speed. That is true of most
  of a language model's kernels too: the card spends its time waiting for memory.
- **Try a different block size** (`//! block 32`, `//! block 1024`) and watch the time and the picture (F3).

When it passes, F6 opens the lesson list.
