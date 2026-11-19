// Lesson 5: unpack 4-bit weights (Q4_K), repacked into separate arrays for the card.
//
// Each thread unpacks one chunk of 32 weights into 32 floats. Chunk c of row r is chunk number
// i = r * n_chunk + c, and its 32 floats go to out[i * 32 .. i * 32 + 31].
//
// The layout (the top of code/core/cuda/qmatvec.cu says the same):
//   qs[i]   16 bytes (one uint4): byte k holds weight k in its low 4 bits and weight k + 16 in its high 4 bits
//   sm[i]   2 bytes: the chunk's scale in the low byte and its min in the high byte (each 0..63)
//   dd[i / 8]  4 bytes, shared by 8 chunks (256 weights): an fp16 d in the low half, an fp16 dmin in the high half
//
//   weight = d * scale * q  -  dmin * min        (q = the 4-bit number, 0..15)
//! block 256

#include <cuda_fp16.h>

extern "C" __global__ void unpack_q4k(const uint4* qs, const unsigned short* sm, const unsigned int* dd,
                                      float* out, int n_rows, int n_chunk)
{

}
