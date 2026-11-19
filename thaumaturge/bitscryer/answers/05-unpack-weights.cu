//! block 256
#include <cuda_fp16.h>
extern "C" __global__ void unpack_q4k(const uint4* qs, const unsigned short* sm, const unsigned int* dd,
                                      float* out, int n_rows, int n_chunk)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * n_chunk) return;
    uint4 q = qs[i];
    unsigned s = sm[i], w = dd[i / 8];
    float scale = __fmul_rn(__half2float(__ushort_as_half((unsigned short)(w & 0xFFFF))), (float)(s & 0xFF));
    float mn = __fmul_rn(__half2float(__ushort_as_half((unsigned short)(w >> 16))), (float)(s >> 8));
    unsigned words[4] = {q.x, q.y, q.z, q.w};
    float v[32];
#pragma unroll
    for (int k = 0; k < 16; k++) {
        unsigned b = (words[k / 4] >> (8 * (k % 4))) & 0xFF;
        v[k] = __fsub_rn(__fmul_rn(scale, (float)(b & 15)), mn);
        v[k + 16] = __fsub_rn(__fmul_rn(scale, (float)(b >> 4)), mn);
    }
    float4* o = reinterpret_cast<float4*>(out + (size_t)i * 32);
#pragma unroll
    for (int e = 0; e < 8; e++) o[e] = make_float4(v[4 * e], v[4 * e + 1], v[4 * e + 2], v[4 * e + 3]);
}
