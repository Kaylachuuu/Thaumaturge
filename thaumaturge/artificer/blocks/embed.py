from thaumaturge.artificer.block import Block, nn, start_weights


class Embed(Block):
    """Turns each word's number into a row of `width` numbers: a lookup in a table the model learns."""
    takes, gives = "tokens", "width"

    def setup(self, width, vocab):
        self.table = nn.Embedding(vocab, width)        # one row per word in the vocabulary
        start_weights(self)

    def forward(self, tokens):
        return self.table(tokens)
