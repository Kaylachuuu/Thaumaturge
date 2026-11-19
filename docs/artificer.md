# Artificer (planned)

A text-mode builder for whole models, from snap-together blocks: the stack of blocks on the right, the chosen
block's code in the editor on the left (the same editor as BitScryer's). Drop blocks in, edit their code,
rearrange them.

The plan, a step at a time:

1. **Shape.** A block is a small file with a fixed plug (width in, width out); a model is a stack file listing
   blocks in order, with repeats. A check builds it on the CPU with a tiny pass: do the plugs fit, how many
   weights, how much card memory in fp16.
2. **Train.** Run a stack on a spare card, with a live view of each layer.
3. **Own kernels.** A block whose inside is a kernel written in BitScryer.

Nothing is built yet.
