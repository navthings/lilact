import mlx.core as mx
import mlx.nn as nn

from activations import MLP

SIZES = {
    "s20": dict(d=384, layers=6, heads=6),
    "s100": dict(d=768, layers=12, heads=12),
}


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = nn.RoPE(self.head_dim, base=10000)

    def __call__(self, x):
        B, T, D = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3) for t in (q, k, v))
        q, k = self.rope(q), self.rope(k)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask="causal")
        return self.o(out.transpose(0, 2, 1, 3).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, act: str, **act_kw):
        super().__init__()
        self.norm1 = nn.RMSNorm(d, eps=1e-5)
        self.attn = Attention(d, heads)
        self.norm2 = nn.RMSNorm(d, eps=1e-5)
        self.mlp = MLP(d, act, **act_kw)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Model(nn.Module):
    def __init__(self, vocab: int, d: int, layers: int, heads: int, act: str, **act_kw):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = [Block(d, heads, act, **act_kw) for _ in range(layers)]
        self.norm = nn.RMSNorm(d, eps=1e-5)

    def __call__(self, ids):
        h = self.embed(ids)
        for b in self.blocks:
            h = b(h)
        return self.embed.as_linear(self.norm(h))

    def mlp_hiddens(self, ids):
        # same forward pass, but returns each layer's post-activation MLP hidden for diagnostics
        h = self.embed(ids)
        out = []
        for b in self.blocks:
            a = h + b.attn(b.norm1(h))
            hid = b.mlp.hidden(b.norm2(a))
            out.append(hid)
            h = a + b.mlp.down(hid)
        return out


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")
