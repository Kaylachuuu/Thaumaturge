from thaumaturge.artificer.block import Block, F, nn, torch, start_weights


class NAME(Block):
    """What this block does, in one line (the block list shows it)."""
    takes, gives = "width", "width"

    def setup(self, width):
        self.layer = nn.Linear(width, width, bias=False)
        start_weights(self)

    def forward(self, x):
        return self.layer(x)
