//! block 16
#define TILE 16
extern "C" __global__ void matmul_plain(const float* A, const float* B, float* C, int M, int N, int K)
{
    int col = blockIdx.x * TILE + threadIdx.x, row = blockIdx.y * TILE + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
extern "C" __global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int col = blockIdx.x * TILE + threadIdx.x, row = blockIdx.y * TILE + threadIdx.y;
    float acc = 0.f;
    for (int t = 0; t < K; t += TILE) {
        As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}
