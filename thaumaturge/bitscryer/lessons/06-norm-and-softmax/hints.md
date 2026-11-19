The norm, step one: each thread adds up the squares of its own values.
```
const half* row = x + (size_t)blockIdx.x * n;
float ss = 0.f;
for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = __half2float(row[i]); ss += v * v; }
```
Then the block's threads must add their `ss` together: lesson 2's tree in shared memory works.
---
A block-wide sum that every thread gets back (blockDim.x a multiple of 32, at most 1024):
```
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);   // within the warp
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < blockDim.x / 32; i++) t += sh[i];
    return t;
}
```
Put it above the kernels (without `extern "C"`), and a `block_max` the same way with `fmaxf` for softmax.
---
The rest of the norm:
```
float r = rsqrtf(block_sum(ss) / n + eps);
for (int i = threadIdx.x; i < n; i += blockDim.x)
    y[(size_t)blockIdx.x * n + i] = __float2half_rn(__half2float(row[i]) * r * __half2float(w[i]));
```
Softmax: the row's max (block_max), then the sum of `expf(s - max)` (block_sum), then each `p = expf(s - max) / sum`.
