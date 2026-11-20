from thaumaturge.artificer.block import Block, F, nn, start_weights


class Out(Block):
    """Scores every word in the vocabulary as the next one. With tie=embed, it reads them back through the
    embedding's table instead of a table of its own (fewer weights, and it often learns better)."""
    takes, gives = "width", "logits"

    def setup(self, width, vocab, tie="embed"):
        self.tie = tie
        self.lent = []                                 # the borrowed table, kept out of this block's own weights
        self.table = None
        if not tie:
            self.table = nn.Linear(width, vocab, bias=False)
            start_weights(self)

    def borrow(self, others):
        if self.tie:
            if self.tie not in others:
                raise ValueError(f"tie={self.tie}: there is no {self.tie} block to borrow from (tie= turns it off)")
            self.lent = [others[self.tie].table]

    def shares(self):
        return self.tie or None

    def forward(self, x):
        weight = self.lent[0].weight if self.tie else self.table.weight
        return F.linear(x, weight)
