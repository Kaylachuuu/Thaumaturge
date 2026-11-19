// Lesson 2: sum a list.  *out = x[0] + x[1] + ... + x[n-1]
//
// *out starts at 0. One thread per value, as in lesson 1, but now the threads have to work together:
// each block adds up its own 256 values, and then the blocks' sums are added together into *out.
//! block 256

#define BLOCK 256      // keep this the same as the block setting above

extern "C" __global__ void sum(const float* x, float* out, int n)
{
    // Shared memory: one small array for the whole block, much faster than the card's main memory.
    // Every thread of this block sees the same `part`; another block has its own.
    __shared__ float part[BLOCK];

    // 1. Each thread puts its own value into part[threadIdx.x] (0 if it is past the end).

    // 2. Wait until every thread of the block has done step 1.

    // 3. Add the values up, inside the block, in steps.

    // 4. One thread adds the block's sum into *out.
}
