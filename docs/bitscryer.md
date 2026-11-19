# BitScryer

A text-mode IDE for learning to program NVIDIA cards and for going on writing kernels after the lessons are done.
It runs in a terminal (over SSH is fine), inside tmux, so the same command reattaches after a dropped connection.

## What it does

You edit right in its left pane, in its own editor (`thaumaturge/common/editor.py`: selection, undo, copy and
paste, find, auto-indent, CUDA and PTX colours). Each save (Ctrl-S):

1. **builds** your file: CUDA with `nvcc -ptx`, then `ptxas` (hand-written PTX skips the first step), then
   `nvdisasm` for the SASS. Every step's output stays in the folder's `.build/`.
2. **shows** your file, the PTX and the SASS side by side (three columns on a wide screen, Tab between PTX and SASS
   on a narrow one). The cursor on a line of your CUDA lights up the PTX and SASS it became. Registers, shared
   memory and spills come from `ptxas -v`; build errors are marked on your lines.
3. **runs** it on the chosen card with the lesson's test data, through the card's driver directly (libcuda with
   ctypes: no PyTorch in between), in a program of its own with a 30-second limit, so a crash or an endless loop
   only ends that program. Each output is checked against a CPU answer (numpy); the first wrong values are shown.
4. **times** it with the card's own clock (events), and holds it against the card's top speed, worked out from what
   the card says of itself: memory GB/s, fp32 FLOPS, tensor-core FLOPS.
5. **draws** how the launch split up (F3): blocks, warps, idle threads at the end, how many blocks each SM holds
   and what limits it (threads, registers, shared memory), and the waves.

## The lessons

In `thaumaturge/bitscryer/lessons/`, each with a lesson text (F1), hints one at a time only when asked (F2), a
starter file, and a check. Your copies live in `~/bitscryer/lessons/` and are never written over.

1. Add two lists: threads, blocks, warps, the bounds check.
2. Sum a list: shared memory, `__syncthreads`, atomics, warp shuffles.
3. Copy at full speed: coalescing, float4, grid-stride loops, the bandwidth ceiling.
4. Lesson 1 in hand-written PTX.
5. Unpack 4-bit weights: Q4_K, as language-model engines store them.
6. Norm and softmax: RMSNorm and softmax, one block per row.
7. Matrix multiply, plain then tiled: arithmetic intensity, shared-memory tiles, bank conflicts.
8. Tensor cores with wmma, and what a real packed-weight kernel adds. Needs a card with tensor cores.

## After the lessons

`bitscryer new FOLDER` starts a project: a kernel and a `check.py` (the same format as the lessons'), and
`bitscryer FOLDER` opens one. A folder with only a `.cu` or `.ptx` file is built and shown, not run.
`bitscryer --selftest` runs every lesson's answer on the card, to show BitScryer itself works there.

## Which cards

`thaumaturge/bitscryer/cards.py`, checked again before every run. With no settings, any card is used unless another
program has it open. A machine whose cards do other work says more in `/etc/thaumaturge/bitscryer.json` (see
`thaumaturge/bitscryer/settings.py` and `examples/bitscryer.json`):

- **never**: cards never used, by UUID, or by the card a systemd unit is pinned to; if that unit can't be read,
  every card over a given size is refused instead.
- **busy**: cards lent out at times, refused while a lock file is held or a systemd unit is running.

A run sees only its own card (CUDA_VISIBLE_DEVICES is that card's UUID). Builds and runs can be kept to some cores
(`cpus`) at a lower priority (`nice`), and BitScryer refuses to run as root.

Writing SASS by hand is not part of it: NVIDIA publishes no assembler for it.
