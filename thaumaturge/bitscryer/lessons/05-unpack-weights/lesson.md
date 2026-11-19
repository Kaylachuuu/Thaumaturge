# Lesson 5: Unpack 4-bit weights

## The idea

A language model with 12 billion weights would take 24 GB as fp16. On the card it can take about 7 GB, because
most weights are stored in **4 bits** each, in a format called Q4_K (from llama.cpp's file format; here it is
repacked into separate arrays so the card can read it fast).

The trick is that weights near each other have similar sizes. So a group of 32 weights (a *chunk*) shares one
small scale and one small offset, and each weight only stores where it falls between them, as a number 0 to 15:

```
weight = d * scale * q  -  dmin * min
```

- `q` is the weight's own 4 bits.
- `scale` and `min` are the chunk's (6 bits each, 0..63).
- `d` and `dmin` are fp16 numbers shared by 8 chunks (256 weights), setting the overall size.

So every 32 weights cost 16 bytes of `q`, 2 bytes of scale and min, and half a byte of `d`: about 4.6 bits each.

This is the real format real engines use. A function like the one you're writing runs for most of the model's
weights, for every word it says.

## Your task

Fill in `unpack_q4k`: one thread per chunk, 32 floats out.

New skills:

- Taking bits apart: `(byte >> 4) & 15`, `x & 0xFF`, `x >> 16`.
- Reading fp16 bits as a number: `__half2float(__ushort_as_half((unsigned short)bits))`
  (from `cuda_fp16.h`, included at the top).
- Reading a `uint4`: one 16-byte load, then its `.x .y .z .w` are four 32-bit words of 4 bytes each.

The check allows for a tiny rounding difference (see below).

## What to look at

- **Bit for bit.** The check also says what share of the values came out *exactly* as the CPU's. If you write
  `scale * q - mn`, the compiler fuses the multiply and the subtract into one `FFMA` instruction, which rounds
  once instead of twice. Find the `FFMA` in the SASS. Here it still comes out bit for bit, and working out why is
  a good puzzle for an assembly person: count the bits. `d` is fp16 (11 significant bits), `scale` is 6 bits, `q`
  is 4: so `d * scale * q` needs at most 21 bits, and a float holds 24. The product is exact, so there is nothing
  for the second rounding to do. A careful engine still writes `__fsub_rn(__fmul_rn(scale, q), mn)` (the `_rn`
  forms are never fused), because in other formats (Q6_K) and other kernels the same shortcut does change the last
  bit, and an engine that must give exactly PyTorch's numbers can't have that.
- **The speed.** Each thread reads 18.5 bytes and writes 128. Where does the time go? Writing 32 floats one by
  one, or as eight `float4` stores (`STG.E.128`)? Try both and compare.
- **A trick fast kernels use.** 4 bits can become a float with no conversion instruction at all: OR them into the
  low bits of 1.0's pattern (`0x3F800000 | (q << 19)` gives 1 + q/16), then subtract 1 and multiply by 16. Try it
  once yours works, and see what changes in the SASS.
