// Lesson 3: copy memory as fast as the card can.  dst[i] = src[i]
//
// Two kernels this time. BitScryer runs and times both, so you can see the difference.
//! block 256

// One float per thread (n of them).
extern "C" __global__ void copy_floats(const float* src, float* dst, int n)
{

}

// Four floats per thread, as one 16-byte float4 (n4 = n / 4 of them).
extern "C" __global__ void copy4(const float4* src, float4* dst, int n4)
{

}
