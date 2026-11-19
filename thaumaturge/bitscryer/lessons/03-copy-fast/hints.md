Both kernels are lesson 1 with one array fewer:
```
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) dst[i] = src[i];
```
and the same with `n4` for `copy4`: a float4 copies like any other value.
---
A grid-stride loop, for when `//! grid` gives fewer threads than items:
```
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
    dst[i] = src[i];
```
