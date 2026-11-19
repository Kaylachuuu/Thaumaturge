# Lesson 6: Norm and softmax

## The idea

Between a language model's big matrix products, every token goes through a handful of small steps. Two of the most important:

**RMSNorm** keeps the numbers in each token's row a steady size. Gemma does it as:

```
r = 1 / sqrt( (x0² + x1² + ... + x3839²) / 3840 + eps )
y[i] = x[i] * r * w[i]
```

Each row is one token, 3,840 values wide (the width of a 12B model such as Gemma 3 12B), and `w` is a learned weight per column.

**Softmax** turns a row of attention scores into weights that add to 1: bigger scores get much bigger weights.

```
p[i] = exp(s[i] - max) / sum of exp(s[j] - max)
```

Taking away the row's largest score first changes nothing mathematically, but without it `exp()` of a big score
overflows to infinity. On fp16-only cards like the V100 this kind of care is everywhere in an engine.

Both are **one row per block**: a block's threads share a row, each taking every 256th value (thread t takes
t, t + 256, t + 512, ...), and then they combine their parts (a sum for the norm; a max and then a sum for
softmax), which is lesson 2 inside a block.

## Your task

Fill in both kernels. They are checked and timed separately, so you can do one at a time; the one you haven't
written shows as wrong.

New pieces:

- `half` values: `__half2float(h)` to fp32, `__float2half_rn(f)` back.
- `rsqrtf(v)` is 1 / sqrt(v) in one instruction; `expf(v)`, `fmaxf(a, b)`.
- `__shfl_xor_sync(0xffffffff, v, offset)`: trade a value with another lane of the warp, no shared memory needed.

## What to look at

- Our real version is `norm_step` in code/core/cuda/step.cu, with `block_sum` and `warp_sum` above it. It does a
  whole chain of steps (norm, add to the stream, scale, norm again) in one launch, because the launches themselves
  were costing more than the arithmetic. Compare its shape with yours.
- Try `expf` against `__expf` (a faster, rougher version): find each in the SASS (`MUFU.EX2`) and see whether the
  check still passes.
- These rows are small, so the timing may sit well under the card's top speed: one block per row leaves only 1,024
  blocks, and each must wait for its own reads before it can combine. Try a bigger and smaller `//! block.softmax`.
