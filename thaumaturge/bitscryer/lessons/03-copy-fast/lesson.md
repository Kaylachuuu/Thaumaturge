# Lesson 3: Copy at full speed

## The idea

Most of what a language model does on the card is limited by **memory**, not arithmetic: to make one word, the
card reads all of its weights (7 GB for a 12B model in 4 bits) once, and does only a couple of sums with each byte. So the first skill is
moving bytes as fast as the card allows.

The card's top speed is set by its memory: the T1000 moves about 160 GB/s, a V100 about 900 GB/s. The
toolbox works that out from the card itself and shows how close you came. No real copy gets to 100%; 80 to 90%
is excellent.

What decides the speed:

- **Coalescing.** When the 32 threads of a warp read 32 neighbouring floats, the card fetches them as a few wide
  pieces (128 bytes at once). When they read scattered addresses, each one costs a fetch of its own. Neighbouring
  threads should touch neighbouring memory.
- **Wide loads.** A `float4` is four floats read in one 16-byte instruction (`LDG.E.128` in the SASS instead of
  `LDG.E`). Fewer instructions for the same bytes leaves the card more room to keep many reads in flight.
- **Enough reads in flight.** Memory takes hundreds of clock cycles to answer. The card hides that by having many
  warps waiting at once; a launch with too few threads can't keep the memory busy.

## Your task

1. Fill in `copy_floats`: the same shape as lesson 1.
2. Fill in `copy4`: the same, with `float4`. One line copies all four.

Both have a goal shown beside the timing (★ when you reach it). Then experiment:

- `//! grid 56` (four blocks per SM on the T1000) with a **grid-stride loop**: each thread copies item `i`, then
  `i + the whole grid's threads`, and so on. Our own kernels are often written this way, sized to fill the card
  exactly once. Is it faster or slower than one thread per item?
- Block sizes from 64 to 1024.

## What to look at

In the SASS, compare the two kernels' loads: `LDG.E` against `LDG.E.128`. Count the instructions per byte
copied.
