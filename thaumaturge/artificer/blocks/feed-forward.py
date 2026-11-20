from thaumaturge.artificer.block import Block, F, math, nn, start_weights


class FeedForward(Block):
    """Each word on its own, through a wider layer and back: where much of what a model knows is kept."""
    takes, gives = "width", "width"

    def setup(self, width, wider=4, repeats=1):
        self.up = nn.Linear(width, wider * width, bias=False)
        self.out = nn.Linear(wider * width, width, bias=False)
        start_weights(self)
        nn.init.normal_(self.out.weight, std=0.02 / math.sqrt(2 * repeats))

    def forward(self, x):
        return self.out(F.gelu(self.up(x), approximate="tanh"))
