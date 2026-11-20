"""What every block is built on. A block is one small file in a model's blocks/ folder, with one class in it:

    from thaumaturge.artificer.block import Block, nn, F, torch

    class Norm(Block):
        \"\"\"The first line here is what the block list shows.\"\"\"
        takes, gives = "width", "width"          # its plugs: what goes in, what comes out

        def setup(self, width, eps=1e-6):         # settings, by name, from the stack file
            self.weight = nn.Parameter(torch.ones(width))
            self.eps = eps

        def forward(self, x):                     # x: (batch, words, width)
            ...

Plugs are one of:

    tokens   the words as numbers, (batch, words): what a model takes in
    width    a row of `width` numbers for every word, (batch, words, width): what blocks pass along
    logits   a score for every word in the vocabulary, for every place, (batch, words, vocab): what a model gives

Two blocks fit together when what the first gives is what the second takes.

setup() asks for the settings it needs by name. Each comes from the block's own line in the stack file
(`attention heads=8`), else from the settings at the top (`heads = 12`). One more is always there to ask for:
`repeats`, how many times over the block appears (12 inside `repeat 12`), which some blocks use to start their
weights smaller so that many of them added up don't grow too big.

A block that uses another block's weights (the output reading words back through the embedding's table) says so
in borrow(): it is given the first block of each name, after every block is made.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

PLUGS = ("tokens", "width", "logits")

__all__ = ["Block", "PLUGS", "torch", "nn", "F", "math", "start_weights"]


class Block(nn.Module):
    takes = "width"
    gives = "width"

    def __init__(self, **settings):
        super().__init__()
        self.setup(**settings)

    def setup(self) -> None:
        pass

    def borrow(self, others: dict) -> None:
        """Called once every block is made: `others` maps a block's name to the first block of that name."""

    def shares(self) -> str | None:
        """The name of the block whose weights this one uses, if it borrowed any."""
        return None


def start_weights(module: nn.Module, std: float = 0.02) -> None:
    """Every Linear and Embedding inside `module` started small and random (normal, std 0.02, as GPT-2 did), biases
    at zero. PyTorch's own starting values suit models without this many layers less well."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=std)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
