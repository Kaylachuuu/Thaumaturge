// A kernel of your own. Save and BitScryer builds it; check.py (beside it) says how to run and check it.
//! block 256

extern "C" __global__ void scale(const float* x, float* y, float k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = k * x[i];
}
