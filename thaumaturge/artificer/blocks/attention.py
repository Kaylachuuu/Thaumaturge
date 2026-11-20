from thaumaturge.artificer.block import Block, F, math, nn, torch, start_weights


def turn(x, start=0):
    """Rotary positions: each pair of numbers in a head is turned by an angle that grows with the word's place, so
    when two words are compared, how far apart they are shows in the result."""
    words, dim = x.shape[-2], x.shape[-1]
    speeds = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = torch.outer(torch.arange(start, start + words, device=x.device, dtype=torch.float32), speeds)
    cos, sin = angles.cos(), angles.sin()
    a, b = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2).type_as(x)


class Attention(Block):
    """Each word looks back at the words before it and takes in what it finds useful (causal self-attention)."""
    takes, gives = "width", "width"

    def setup(self, width, heads, repeats=1):
        if width % heads:
            raise ValueError(f"width ({width}) must divide evenly into heads ({heads})")
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)   # each word's question, key and value, all at once
        self.out = nn.Linear(width, width, bias=False)
        start_weights(self)
        nn.init.normal_(self.out.weight, std=0.02 / math.sqrt(2 * repeats))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k = turn(q), turn(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # is_causal: never a later word
        return self.out(y.transpose(1, 2).reshape(B, T, C))
