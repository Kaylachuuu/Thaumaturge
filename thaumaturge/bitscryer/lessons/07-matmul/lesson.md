# Lesson 7: Matrix multiply, plain then tiled

## The idea

Almost all of a language model's arithmetic is matrix multiplies: every token's row of 3,840 numbers times weight matrices.
This is where a GPU earns its keep, and the lesson where the gap between "right" and "fast" is widest.

Each value of C is a row of A times a column of B: `C[r][c] = sum over k of A[r][k] * B[k][c]`.

**The plain way:** one thread per value of C, reading its own row and column from main memory. Correct, but
every value of A is read N times over and every value of B M times: 2 GB of reading for 12 MB of matrices. Memory
can't keep up, and the arithmetic units sit waiting.

**The tiled way:** a block of 16 x 16 threads computes a 16 x 16 tile of C together. They walk along K a tile at
a time: all 256 threads copy one 16 x 16 piece of A and one of B into shared memory (one value each), then each
thread does 16 multiply-adds from shared memory. Every value read from main memory is now used 16 times.

The rule behind it: **arithmetic intensity**, the flops done per byte read. A copy does 0; a plain matmul about
1/4; a tiled one about 16/4. The card can do roughly 15 flops (T1000) to 17 flops (V100) in the time it reads
one byte, so to be limited by arithmetic instead of memory, a kernel needs at least that many.

## Your task

1. `matmul_plain` first. Mind the order of the loops' indexes: A is read along a row, B down a column.
2. `matmul_tiled`. The goal is 25% of the card's fp32 top speed.

Both are timed in flops held against the card's fp32 top speed.

## What to look at

- The difference in time between the two. Then try `//! block 32` and `#define TILE 32` (32 x 32 = 1,024
  threads, the most a block can have).
- In the SASS of the tiled kernel's inner loop: `LDS` (shared memory loads) and `FFMA`. How many loads per FFMA?
  The next step up (each thread computing 4 or 8 values of C, keeping them in registers) is how real kernels
  reach 70%+, by lowering that ratio.
- **Bank conflicts:** shared memory is split into 32 banks; if two lanes of a warp read different addresses in
  the same bank, they take turns. Reading `Bs[k][threadIdx.x]` is free of them; `As[threadIdx.x][k]` would not be.
