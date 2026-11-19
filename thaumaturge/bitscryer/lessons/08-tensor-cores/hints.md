Which tile:
```
int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
int tiles_across = N / 16;
int tr = warp / tiles_across, tc = warp % tiles_across;
if (tr * 16 >= M) return;
```
(All 32 threads of the warp get the same numbers, which is what wmma needs: every call is made by the whole warp.)
---
The loop:
```
for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a, A + (size_t)tr * 16 * K + k, K);
    wmma::load_matrix_sync(b, B + (size_t)k * N + tc * 16, N);
    wmma::mma_sync(acc, a, b, acc);
}
wmma::store_matrix_sync(C + (size_t)tr * 16 * N + tc * 16, acc, N, wmma::mem_row_major);
```
