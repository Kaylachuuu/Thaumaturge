//! block 256
#include <cuda_fp16.h>
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < blockDim.x / 32; i++) t += sh[i];
    return t;
}
__device__ float block_max(float v) {
    __shared__ float sh[32];
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int i = 0; i < blockDim.x / 32; i++) t = fmaxf(t, sh[i]);
    return t;
}
extern "C" __global__ void rms_norm(const half* x, const half* w, half* y, int n, float eps)
{
    const half* row = x + (size_t)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = __half2float(row[i]); ss += v * v; }
    float r = rsqrtf(block_sum(ss) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[(size_t)blockIdx.x * n + i] = __float2half_rn(__half2float(row[i]) * r * __half2float(w[i]));
}
extern "C" __global__ void softmax(const float* s, float* p, int n)
{
    const float* row = s + (size_t)blockIdx.x * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, row[i]);
    m = block_max(m);
    float t = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) t += expf(row[i] - m);
    t = block_sum(t);
    for (int i = threadIdx.x; i < n; i += blockDim.x) p[(size_t)blockIdx.x * n + i] = expf(row[i] - m) / t;
}
