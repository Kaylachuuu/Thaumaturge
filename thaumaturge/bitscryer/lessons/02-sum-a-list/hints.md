Step 1 and 2:
```
int i = blockIdx.x * blockDim.x + threadIdx.x;
part[threadIdx.x] = (i < n) ? x[i] : 0.0f;
__syncthreads();
```
---
Step 3 is a loop that halves the number of working threads each time, with a barrier inside it (every thread
reaches it, so it's safe):
```
for (int s = BLOCK / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
    __syncthreads();
}
```
---
Step 4: after the loop the block's sum is in part[0].
```
if (threadIdx.x == 0) atomicAdd(out, part[0]);
```
