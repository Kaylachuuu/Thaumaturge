# Lesson 4: The same, in PTX

## The idea

When you build CUDA, it goes through two steps:

```
kernel.cu --nvcc--> PTX --ptxas--> SASS (the card's real machine code)
```

**PTX** is NVIDIA's published assembly language. It is the lowest level you're officially allowed to write: it
stays the same from card to card, and `ptxas` turns it into the SASS of the particular card. Think of it like a
portable assembly with as many registers as you care to declare.

**SASS** is what really runs. NVIDIA doesn't publish an assembler for it, so you can read it (the right-hand pane)
but not officially write it. Comparing your PTX with the SASS shows you what `ptxas` decided: which registers,
what order, what it combined.

A PTX instruction reads like `opcode.type destination, sources;`:

```
ld.param.u64        %rd1, [a];         // a 64-bit parameter into %rd1
mov.u32             %r1, %ctaid.x;     // blockIdx.x into %r1
mad.lo.s32          %r4, %r1, %r2, %r3;// %r4 = %r1 * %r2 + %r3 (the low 32 bits)
setp.ge.s32         %p1, %r4, %r5;     // %p1 = (%r4 >= %r5)
@%p1 bra            DONE;              // jump if %p1 is true (any instruction can have an @predicate)
mul.wide.s32        %rd7, %r4, 4;      // 32-bit times 32-bit, into a 64-bit result
add.s64             %rd8, %rd4, %rd7;  // address arithmetic
ld.global.f32       %f1, [%rd8];       // load a float from main memory
add.f32             %f3, %f1, %f2;
st.global.f32       [%rd10], %f3;
```

Coming from x86 or ARM assembly, the big differences are: no stack to speak of, every thread runs this same code
at once, and any instruction can be made conditional with a predicate.

## Your task

Write the body of `add` in PTX, following the five steps in the file. The check is the same as lesson 1.

The PTX you write lights up nothing in the SASS pane (hand-written PTX carries no line marks), but it's short
enough to follow by eye.

## What to look at

- Open lesson 1's file in your mind, or literally: `.build/kernel.ptx` in lesson 1's folder
  (~/bitscryer/lessons/01-add-lists/.build/) is the PTX nvcc wrote for your CUDA. How close is yours?
- In the SASS, ptxas changed your `mul.wide` + `add` into one `IMAD.WIDE`, and the parameter loads became
  reads of the constant bank `c[0x0][0x160]`. The card keeps a kernel's parameters there.
- Try `.target sm_75` and save: on the V100 (sm_70) it won't load any more, which is what `.target` means.
