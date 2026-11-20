from thaumaturge.artificer.block import Block, nn, torch


class Norm(Block):
    """Scales each word's row so its numbers have a steady size (RMS norm), then by a learned amount per number."""
    takes, gives = "width", "width"

    def setup(self, width, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps                                 # keeps it from dividing by zero

    def forward(self, x):
        xf = x.float()                                 # in fp32: squares of fp16 numbers overflow easily
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * rms).type_as(x) * self.weight
