//! block 256
extern "C" __global__ void copy_floats(const float* src, float* dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
extern "C" __global__ void copy4(const float4* src, float4* dst, int n4)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}
