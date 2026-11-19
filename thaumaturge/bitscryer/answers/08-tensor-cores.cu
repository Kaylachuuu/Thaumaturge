//! block 128
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void mm_wmma(const half* A, const half* B, float* C, int M, int N, int K)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tiles_across = N / 16;
    int tr = warp / tiles_across, tc = warp % tiles_across;
    if (tr * 16 >= M) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + (size_t)tr * 16 * K + k, K);
        wmma::load_matrix_sync(b, B + (size_t)k * N + tc * 16, N);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(C + (size_t)tr * 16 * N + tc * 16, acc, N, wmma::mem_row_major);
}
