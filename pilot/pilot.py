import json
import math
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

text = open("tiny.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = mx.array([stoi[c] for c in text], dtype=mx.int32)
n = int(0.9 * len(text))
train_data, val_data = data[:n], data[n:]

D, LAYERS, HEADS, SEQ, BATCH = 256, 6, 4, 256, 32
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
WARMUP, LR = 150, 1e-3
EVAL_EVERY, EVAL_BATCHES = 250, 20
GLU_FF = 680  # ~8/3 * D so GLU and plain 4*D MLPs have the same param count


def batch(src, key):
    ix = mx.random.randint(0, src.size - SEQ - 1, (BATCH,), key=key)
    idx = ix[:, None] + mx.arange(SEQ + 1)[None, :]
    chunk = src[idx]
    return chunk[:, :-1], chunk[:, 1:]


class SiLUSq(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = mx.zeros((dim,))

    def __call__(self, x):
        s = nn.silu(x)
        return s + self.a * s * s


class PlainMLP(nn.Module):
    def __init__(self, act, d_ff):
        super().__init__()
        self.up = nn.Linear(D, d_ff, bias=False)
        self.down = nn.Linear(d_ff, D, bias=False)
        self.act = act

    def __call__(self, x):
        return self.down(self.act(self.up(x)))


class GLUMLP(nn.Module):
    def __init__(self, act, d_ff):
        super().__init__()
        self.w = nn.Linear(D, d_ff, bias=False)
        self.v = nn.Linear(D, d_ff, bias=False)
        self.down = nn.Linear(d_ff, D, bias=False)
        self.act = act

    def __call__(self, x):
        return self.down(self.act(self.w(x)) * self.v(x))


def relu2(x):
    return mx.square(nn.relu(x))


def make_mlp(kind):
    return {
        "gelu": lambda: PlainMLP(nn.gelu, 4 * D),
        "relu2": lambda: PlainMLP(relu2, 4 * D),
        "silusq": lambda: PlainMLP(SiLUSq(4 * D), 4 * D),
        "swiglu": lambda: GLUMLP(nn.silu, GLU_FF),
        "silusq_glu": lambda: GLUMLP(SiLUSq(GLU_FF), GLU_FF),
    }[kind]()


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.mlp = make_mlp(kind)

    def __call__(self, x):
        B, T, _ = x.shape
        q, k, v = mx.split(self.qkv(self.n1(x)), 3, axis=-1)
        q, k, v = (t.reshape(B, T, HEADS, D // HEADS).transpose(0, 2, 1, 3) for t in (q, k, v))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D // HEADS) ** -0.5, mask="causal")
        x = x + self.proj(a.transpose(0, 2, 1, 3).reshape(B, T, D))
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.tok = nn.Embedding(len(chars), D)
        self.pos = nn.Embedding(SEQ, D)
        self.blocks = [Block(kind) for _ in range(LAYERS)]
        self.norm = nn.RMSNorm(D)
        self.head = nn.Linear(D, len(chars), bias=False)

    def __call__(self, x):
        h = self.tok(x) + self.pos(mx.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


def evaluate(model):
    key = mx.random.key(1234)
    total = 0.0
    for _ in range(EVAL_BATCHES):
        key, sub = mx.random.split(key)
        total += loss_fn(model, *batch(val_data, sub)).item()
    return total / EVAL_BATCHES


def run(kind, seed):
    mx.random.seed(seed)
    model = Model(kind)
    mx.eval(model.parameters())

    def schedule():
        warm = optim.linear_schedule(1e-7, LR, WARMUP)
        cos = optim.cosine_decay(LR, STEPS - WARMUP, end=LR * 0.1)
        return optim.join_schedules([warm, cos], [WARMUP])

    # weight decay only on matrices, never on norms or the learned a
    opt = optim.MultiOptimizer(
        [optim.AdamW(schedule(), betas=[0.9, 0.95], weight_decay=0.1),
         optim.AdamW(schedule(), betas=[0.9, 0.95], weight_decay=0.0)],
        [lambda path, w: w.ndim >= 2],
    )
    state = [model.state, opt.state]

    def train_step(x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    train_step = mx.compile(train_step, inputs=state, outputs=state)

    key = mx.random.key(seed)
    hist, t0 = [], time.time()
    for s in range(STEPS + 1):
        if s % EVAL_EVERY == 0:
            hist.append((s, round(evaluate(model), 4)))
            print(kind, seed, s, hist[-1][1], f"{time.time() - t0:.0f}s", flush=True)
        if s == STEPS:
            break
        key, sub = mx.random.split(key)
        loss = train_step(*batch(train_data, sub))
        mx.eval(state, loss)

    a_stats = [
        (name, round(p.mean().item(), 4), round(p.min().item(), 4), round(p.max().item(), 4))
        for name, p in tree_flatten(model.parameters()) if name.endswith(".a")
    ]
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    return {"kind": kind, "seed": seed, "params": n_params, "val": hist,
            "final": hist[-1][1], "secs": round(time.time() - t0, 1), "a_per_layer": a_stats}


if __name__ == "__main__":
    res = run(sys.argv[1], int(sys.argv[2]))
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
