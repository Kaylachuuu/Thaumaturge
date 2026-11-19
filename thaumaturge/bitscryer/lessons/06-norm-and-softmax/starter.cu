// Lesson 6: two of the small steps between a language model's matrix products. One block per row, for both.
//! block 256

#include <cuda_fp16.h>

// RMSNorm, as Gemma does it: for each row of n values,
//   r = 1 / sqrt(mean of x*x + eps)          (worked out in fp32)
//   y[i] = x[i] * r * w[i]                    (rounded to fp16 at the end)
// Row number blockIdx.x starts at x + blockIdx.x * n.
extern "C" __global__ void rms_norm(const half* x, const half* w, half* y, int n, float eps)
{

}

// Softmax: for each row of n scores,
//   p[i] = exp(s[i] - m) / (sum over j of exp(s[j] - m)),   where m is the row's largest score.
extern "C" __global__ void softmax(const float* s, float* p, int n)
{

}
