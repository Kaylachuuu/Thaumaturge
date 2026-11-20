# Artificer

A text-mode builder for whole models, from snap-together blocks: the stack of blocks on the right, the chosen
block's code in the editor on the left (the same editor as BitScryer's). Add blocks, edit their code, rearrange
them, and check that the model still fits together.

```
 Artificer · tiny-story                                                          ✓ fits · 123.6M weights
╭─ blocks/attention.py ─────────────────────────╮╭─ the stack · tiny-story.stack ──────────────────────────╮
│  17  class Attention(Block):                  ││ embed                                → 768 wide  38.6M │
│  18      """Each word looks back at the words ││ repeat ×12                                       85.0M │
│  19      takes, gives = "width", "width"      ││ │ add (shortcut)                                 28.3M │
│  20                                           ││ │ │ norm                     → 768 wide  9.2K (12×768) │
│  21      def setup(self, width, heads, repeats││▶│ │ attention              → 768 wide  28.3M (12×2.4M) │
│  ...                                          ││ │ add (shortcut)                                 56.6M │
│                                               ││ │ │ norm                     → 768 wide  9.2K (12×768) │
│                                               ││ │ │ feed-forward           → 768 wide  56.6M (12×4.7M) │
│                                               ││ norm                                   → 768 wide  768 │
│                                               ││ out                        → 50304 words  uses embed's │
│                                               ││────────────────────────────────────────────────────────│
│                                               ││ 123.6M weights · to run, in fp16: 247.2 MB             │
│                                               ││ to train, batch 8 × 512 words: about 4.6 GB (rough)    │
╰───────────────────────────────────────────────╯╰────────────────────────────────────────────────────────╯
```

```
artificer                     the model you had open last (at first: tiny-story, made in ~/artificer/)
artificer ~/artificer/mine    a model of your own: a folder with a .stack file and a blocks/ folder
artificer new ~/artificer/x   start a model, from tiny-story's stack and blocks
artificer --check FOLDER      check a model and print what it found
```

## Blocks

A block is one small Python file in the model's `blocks/` folder, with one class in it built on `Block`
(`thaumaturge/artificer/block.py`), written with PyTorch:

```python
from thaumaturge.artificer.block import Block, nn, torch

class Norm(Block):
    """Scales each word's row so its numbers have a steady size."""   # the first line shows in the block list
    takes, gives = "width", "width"                                     # its plugs

    def setup(self, width, eps=1e-6):                                   # its settings, asked for by name
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        ...
```

Its **plugs** say what goes in and what comes out, one of:

- `tokens`: the words as numbers, (batch, words). What a model takes in.
- `width`: a row of `width` numbers for every word, (batch, words, width). What blocks pass along.
- `logits`: a score for every word in the vocabulary, for every place, (batch, words, vocab). What a model gives.

Two blocks fit when what the first gives is what the next takes. You write only `forward`; PyTorch works out the
backward pass that training needs.

The library (`thaumaturge/artificer/blocks/`) has embed, norm, attention, feed-forward and out. Adding one copies it
into the model's `blocks/`, so changing it there never changes the library or another model. `n` makes a new
block from a template.

## The stack file

```
# The small story model
width = 768                    settings any block can ask for, by name
heads = 12
vocab = 50304
context = 512

embed
repeat 12                      what's indented under it, 12 times over, each copy with its own weights
    add                        a shortcut: what goes in comes out again, plus what the blocks inside make of it
        norm
        attention heads=8      a setting for this block alone
    add
        norm
        feed-forward
norm
out
```

The screen changes it for you and writes it back; it stays readable, and `e` opens it as text. A block's
`setup()` can also ask for `repeats`: how many copies of it there are, which some blocks use to start their
weights smaller.

## The check

Every change and every Ctrl-S checks the model, in a program of its own, on the CPU only, with a two-minute limit
(a block stuck in a loop is stopped, and the screen carries on):

1. the stack file reads;
2. every block's file loads (a mistake is marked on its line in the editor);
3. the plugs fit, top to bottom; inside a shortcut the blocks end where they started (width);
4. every block can be made with the settings it asks for;
5. a few words go all the way through, every width block gives back the shape it took, and what comes out is
   (1, words, vocab), with no inf or NaN.

It then counts the weights (shared ones once) and works out the card memory in fp16: to run it (2 bytes a
weight), and roughly to train it (about 16 bytes a weight for the fp16 weights, an fp32 copy, the gradients and
Adam's two averages, plus what each block keeps for the backward pass at the stack's `batch` × `context`).

The starter, tiny-story, is the same model as the small test model it was taken from: built from these blocks, with
the same weights, it gives exactly the same answers.

## Keys

On the stack: ↑ ↓ choose, Enter opens a block's code, `a` add (after the chosen block; into a group at its top),
`n` new block, `d` remove, `c` copy, Shift-↑ ↓ or `[` `]` move (stepping into and out of groups), `g` put inside a
shortcut, `r` repeat (on a repeat: how many times), `x` take a group away keeping what's inside, `=` a block's own
settings, `s` check, `z` undo, `e` the stack file as text, `?` help, `q` quit.

In the code: the editor's keys (as BitScryer's); Ctrl-S saves and checks; Esc goes back to the stack.

F1 help, F2 check, F5 the stack file, F6 code or stack, F10 or Ctrl-Q quit; Esc then a letter does what the letter
does on the stack.

## What comes next

1. **Shape** (built): blocks, the stack file, the check, the screen.
2. **Train.** Run a stack on a card, with a live view of each layer.
3. **Own kernels.** A block whose inside is a kernel written in BitScryer.
