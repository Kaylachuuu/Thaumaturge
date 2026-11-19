// Lesson 7: C = A x B.  A is M x K, B is K x N, C is M x N, all fp32, stored a row at a time ("row-major"):
// A's row r, column c is A[r * K + c].
//
// The blocks are square this time: TILE x TILE threads, (threadIdx.x, threadIdx.y). Thread (x, y) of block
// (blockIdx.x, blockIdx.y) owns ONE value of C: column blockIdx.x * TILE + x, row blockIdx.y * TILE + y.
//! block 16

#define TILE 16        // keep this the same as the block setting above

// The plain way: the thread reads its row of A and its column of B straight from memory.
extern "C" __global__ void matmul_plain(const float* A, const float* B, float* C, int M, int N, int K)
{

}

// The tiled way: the block walks along K one TILE at a time. Each step, its threads together copy a TILE x TILE
// piece of A and of B into shared memory, wait, multiply from shared memory, wait, and move on.
extern "C" __global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

}
