// Lesson 8: C = A x B on the tensor cores.  A (M x K) and B (K x N) are fp16, row-major; C (M x N) is fp32.
//
// Each WARP (not each thread) owns one 16 x 16 tile of C. Warp w of block b has tile t = b * (warps per block) + w;
// the tiles are numbered along C's rows: tile row t / (N / 16), tile column t % (N / 16).
// Needs a card with tensor cores: the V100 or a 2080, not the T1000.
//! block 128

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void mm_wmma(const half* A, const half* B, float* C, int M, int N, int K)
{
    // Which tile is this warp's?

    // A fragment holds a 16 x 16 tile spread over the warp's 32 threads' registers; you never touch it directly.
    //   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    //   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    //   wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    //   wmma::fill_fragment(acc, 0.0f);

    // Walk along K, 16 at a time: load a tile of A and of B, and multiply-add them into acc.
    //   wmma::load_matrix_sync(a, pointer to the tile's first value, the row length of the matrix);
    //   wmma::mma_sync(acc, a, b, acc);

    // Store acc into C:  wmma::store_matrix_sync(pointer, acc, N, wmma::mem_row_major);
}
