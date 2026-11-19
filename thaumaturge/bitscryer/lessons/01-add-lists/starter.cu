// Lesson 1: add two lists.  c[i] = a[i] + b[i]  for every i from 0 to n-1.
//
// Type right here. Save (Ctrl-S) and BitScryer builds it, runs it on the card and checks every value against the CPU.
//
// The line below is a setting for BitScryer, not for the compiler: how many threads go in each block.
// BitScryer launches enough blocks that there is one thread for every value (and a few more).
//! block 256

extern "C" __global__ void add(const float* a, const float* b, float* c, int n)
{
    // Every thread runs this same function. What makes each one different is where it sits:
    //   threadIdx.x   which thread it is inside its block (0 to blockDim.x - 1)
    //   blockIdx.x    which block it is in (0 to gridDim.x - 1)
    //   blockDim.x    how many threads a block has (256 here)
    //
    // Your part: work out which value this thread owns, and add it.
    // (F1 opens the lesson, which says more. F2 for a hint, only if you want one.)

}
