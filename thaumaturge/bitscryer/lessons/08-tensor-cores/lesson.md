# Lesson 8: Tensor cores

## The idea

From the V100 on, NVIDIA cards have **tensor cores**: units that multiply small matrices (16 x 16 tiles of fp16
numbers, adding in fp32) in one go. On the V100 they are eight times faster than the ordinary fp32 cores: about
125 TFLOPS against 15. The T1000's chip was cut down and has none, which is why this lesson only runs on the V100
or a newer card with tensor cores (F4 to choose the card).

They work at the level of a **warp**, not a thread: the 32 threads together hold each tile spread across their
registers (a *fragment*), and one `mma_sync` is the whole warp multiplying two 16 x 16 tiles and adding the result
into a third. CUDA's `wmma` functions are the way in:

```
wmma::fill_fragment(acc, 0.0f);
for (k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a, &A[tile_row * 16 * K + k], K);      // A's tile at (tile_row, k)
    wmma::load_matrix_sync(b, &B[k * N + tile_col * 16], N);      // B's tile at (k, tile_col)
    wmma::mma_sync(acc, a, b, acc);
}
wmma::store_matrix_sync(&C[tile_row * 16 * N + tile_col * 16], acc, N, wmma::mem_row_major);
```

The last number of each load is the **leading dimension**: how far apart, in values, one row of the big matrix
is from the next.

## Your task

Write `mm_wmma`, one warp per tile of C, as the comments in the file lay out. The goal is 15% of the tensor cores'
top speed; this straightforward version reads everything from main memory, so it won't get much further.

## Then: what a real kernel adds

The kernels that do a language model's many-token multiplies (reading a prompt, checking guesses) straight from
packed 4-bit weights are built from exactly these pieces, and a few more. Each is worth trying on yours:

- **Decoding into shared memory.** Each step, every thread unpacks one chunk of 32 weights (lesson 5) into a shared
  tile, and the tokens' matching columns go into another. The warps then `mma_sync` from shared memory, not main
  memory.
- **Several fragments per warp.** A warp keeps one accumulator per 16-token stripe, so each tile of weights it
  loads is used several times.
- **Reading ahead.** Start reading the *next* step's weights from main memory before this step's multiply, so the
  memory and the tensor cores work at the same time.
- **Padding.** Make each shared row 8 halves longer than needed. That shifts each row onto different banks, so the
  tensor cores' loads don't take turns (lesson 7's bank conflicts).
- **Adding up in a fixed order.** When several warps split the work, add their sums in the same order every time,
  so the same call always gives the same numbers, bit for bit. Rounding that changes from run to run makes it
  very hard to check that a change to an engine changed nothing else.

In the SASS of yours, `HMMA` is the tensor-core instruction. On the V100 one `mma_sync` becomes several `HMMA`s;
count them, and compare with a 2080's if you get one (Volta and Turing split the work differently).
