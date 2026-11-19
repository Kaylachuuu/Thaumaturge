Plain:
```
int col = blockIdx.x * TILE + threadIdx.x, row = blockIdx.y * TILE + threadIdx.y;
if (row >= M || col >= N) return;
float acc = 0.f;
for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
C[row * N + col] = acc;
```
---
Tiled: the loop walks K in steps of TILE. In each step, thread (x, y) copies one value of each:
```
for (int t = 0; t < K; t += TILE) {
    As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
    __syncthreads();
    ...
    __syncthreads();
}
```
(The sizes here are all multiples of 16, so no thread needs a bounds check inside the loop.)
---
Between the two barriers:
```
for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
```
The second `__syncthreads()` keeps the next step's copy from writing over a tile someone is still reading.
