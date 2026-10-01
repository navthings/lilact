import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from activations import ACTS, EXTRA_ACTS, GroupGate, RankReLU2
from data import Batches, val_batches
from model import SIZES, Model, loss_fn

# tokens per step = micro_bs * accum * seq; s100 uses smaller micro-batches to stay well under 16GB
DEFAULTS = {
    "s20": dict(tokens=100_000_000, micro_bs=8, accum=4),
    "s100": dict(tokens=300_000_000, micro_bs=4, accum=16),
}


def run_name(a) -> str:
    return f"{a.size}_{a.act}{a.tag}_lr{a.lr:g}_s{a.seed}"


def evaluate(model, batches) -> float:
    return sum(loss_fn(model, x, y).item() for x, y in batches) / len(batches)


def diagnostics(model, x) -> dict:
    layers = []
    for i, hid in enumerate(model.mlp_hiddens(x)):
        flat = hid.reshape(-1, hid.shape[-1])
        layers.append({
            "layer": i,
            "zero_frac": (flat == 0).mean().item(),
            "dead_units_frac": (mx.abs(flat).max(axis=0) == 0).mean().item(),
            "max_abs": mx.abs(flat).max().item(),
        })
    learned = []
    for i, b in enumerate(model.blocks):
        act = getattr(b.mlp, "act", None)
        if isinstance(act, (GroupGate, RankReLU2)):
            learned.append({"layer": i, **act.stats()})
    return {"layers": layers, "learned": learned}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--act", required=True, choices=ACTS + EXTRA_ACTS)
    p.add_argument("--size", default="s20", choices=list(SIZES))
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokens", type=int)
    p.add_argument("--micro_bs", type=int)
    p.add_argument("--accum", type=int)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--warmup_frac", type=float, default=0.02)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--g", type=int, default=8)
    p.add_argument("--beta", type=float, default=1.5)
    p.add_argument("--rank_stop_grad", action="store_true")
    p.add_argument("--tag", default="", help="suffix for ablation runs, e.g. _g32")
    p.add_argument("--data", default="data/tinystories")
    p.add_argument("--out", default="runs")
    a = p.parse_args()
    for k, v in DEFAULTS[a.size].items():
        if getattr(a, k) is None:
            setattr(a, k, v)

    out_path = Path(a.out) / f"{run_name(a)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    vocab = int((data / "vocab_size.txt").read_text())

    mx.random.seed(a.seed)
    model = Model(vocab, act=a.act, g=a.g, beta=a.beta, rank_stop_grad=a.rank_stop_grad, **SIZES[a.size])
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))

    tokens_per_step = a.micro_bs * a.accum * a.seq
    steps = a.tokens // tokens_per_step
    warmup = max(1, min(steps // 10, max(100, int(a.warmup_frac * steps))))
    train = Batches(data / "train.bin", a.seq, a.seed)
    if train.tokens_available() < steps * tokens_per_step:
        raise RuntimeError(f"need {steps * tokens_per_step:,} train tokens, have {train.tokens_available():,}")
    val = val_batches(data / "val.bin", a.seq, a.micro_bs, a.eval_batches)

    def schedule():
        warm = optim.linear_schedule(1e-8, a.lr, warmup)
        cos = optim.cosine_decay(a.lr, steps - warmup, end=a.lr * 0.1)
        return optim.join_schedules([warm, cos], [warmup])

    # weight decay on weight matrices only, never on norms or the activations' own params
    opt = optim.MultiOptimizer(
        [optim.AdamW(schedule(), betas=[0.9, 0.95], weight_decay=0.1),
         optim.AdamW(schedule(), betas=[0.9, 0.95], weight_decay=0.0)],
        [lambda path, w: w.ndim >= 2 and ".act." not in path],
    )
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    # value_and_grad swaps params inside the trace, so state must be both input and output
    micro = mx.compile(lambda x, y: loss_and_grad(model, x, y), inputs=model.state, outputs=model.state)

    print(f"{run_name(a)}: {n_params / 1e6:.1f}M params, {steps} steps, {tokens_per_step} tokens/step")
    history, best, train_time, run_loss, run_n = [], math.inf, 0.0, 0.0, 0
    for step in range(steps + 1):
        if step % a.eval_every == 0 or step == steps:
            v = evaluate(model, val)
            best = min(best, v)
            tps = (step * tokens_per_step / train_time) if train_time else 0.0
            history.append({"step": step, "tokens": step * tokens_per_step, "val": v,
                            "train": run_loss / run_n if run_n else None, "tok_per_s": tps})
            run_loss, run_n = 0.0, 0
            print(f"step {step:6d}  val {v:.4f}  best {best:.4f}  {tps:,.0f} tok/s", flush=True)
            out_path.write_text(json.dumps({"config": vars(a), "params": n_params, "history": history,
                                            "best_val": best, "done": False}, indent=1))
        if step == steps:
            break

        t0 = time.perf_counter()
        grads, loss_sum = None, 0.0
        for _ in range(a.accum):
            loss, g = micro(*train.next(a.micro_bs))
            grads = g if grads is None else tree_map(mx.add, grads, g)
            loss_sum = loss_sum + loss
        grads = tree_map(lambda t: t / a.accum, grads)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.state, opt.state, loss_sum)
        train_time += time.perf_counter() - t0
        loss_val = loss_sum.item() / a.accum
        if not math.isfinite(loss_val):
            raise RuntimeError(f"loss went to {loss_val} at step {step}")
        run_loss += loss_val
        run_n += 1

    result = {
        "config": vars(a),
        "params": n_params,
        "history": history,
        "best_val": best,
        "final_val": history[-1]["val"],
        "tok_per_s": steps * tokens_per_step / train_time,
        "diagnostics": diagnostics(model, val[0][0]),
        "done": True,
    }
    out_path.write_text(json.dumps(result, indent=1))
    print(f"done: best {best:.4f}, final {result['final_val']:.4f}, saved {out_path}")


if __name__ == "__main__":
    main()
