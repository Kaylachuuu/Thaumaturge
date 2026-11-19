Think of the blocks laid end to end: block 0 has values 0 to 255, block 1 has 256 to 511, and so on. So the first
value of block `b` is `b * 256`, that is `blockIdx.x * blockDim.x`.
---
```
int i = blockIdx.x * blockDim.x + threadIdx.x;
```
Now, which threads must not do the addition?
---
Any thread whose `i` is `n` or more is past the end:
```
if (i < n) c[i] = a[i] + b[i];
```
