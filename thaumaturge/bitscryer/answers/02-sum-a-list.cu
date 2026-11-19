//! block 256
#define BLOCK 256
extern "C" __global__ void sum(const float* x, float* out, int n)
{
    __shared__ float part[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    part[threadIdx.x] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, part[0]);
}
