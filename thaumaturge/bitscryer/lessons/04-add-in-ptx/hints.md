Step 1:
```
ld.param.u64        %rd1, [a];
ld.param.u64        %rd2, [b];
ld.param.u64        %rd3, [c];
ld.param.u32        %r5, [n];
cvta.to.global.u64  %rd4, %rd1;
cvta.to.global.u64  %rd5, %rd2;
cvta.to.global.u64  %rd6, %rd3;
```
---
Steps 2 and 3:
```
mov.u32             %r1, %ctaid.x;
mov.u32             %r2, %ntid.x;
mov.u32             %r3, %tid.x;
mad.lo.s32          %r4, %r1, %r2, %r3;
setp.ge.s32         %p1, %r4, %r5;
@%p1 bra            DONE;
```
---
Steps 4 and 5:
```
mul.wide.s32        %rd7, %r4, 4;
add.s64             %rd8, %rd4, %rd7;
add.s64             %rd9, %rd5, %rd7;
add.s64             %rd10, %rd6, %rd7;
ld.global.f32       %f1, [%rd8];
ld.global.f32       %f2, [%rd9];
add.f32             %f3, %f1, %f2;
st.global.f32       [%rd10], %f3;
```
