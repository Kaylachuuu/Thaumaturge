Start with which chunk this thread owns, and stop past the end:
```
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= n_rows * n_chunk) return;
```
Then read the three pieces: `uint4 q = qs[i];`, `unsigned s = sm[i];`, `unsigned w = dd[i / 8];`
---
The two numbers shared by the chunk:
```
float d    = __half2float(__ushort_as_half((unsigned short)(w & 0xFFFF)));
float dmin = __half2float(__ushort_as_half((unsigned short)(w >> 16)));
float scale = d * (float)(s & 0xFF);
float mn    = dmin * (float)(s >> 8);
```
---
The 16 bytes as four words, and byte k of them:
```
unsigned words[4] = {q.x, q.y, q.z, q.w};
for (int k = 0; k < 16; k++) {
    unsigned b = (words[k / 4] >> (8 * (k % 4))) & 0xFF;
    out[i * 32 + k]      = scale * (float)(b & 15) - mn;
    out[i * 32 + k + 16] = scale * (float)(b >> 4) - mn;
}
```
---
For every value bit for bit: `__fsub_rn(__fmul_rn(scale, (float)(b & 15)), mn)`, and the same for the two
numbers shared by the chunk (`__fmul_rn(d, (float)(s & 0xFF))`).
