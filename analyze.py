import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from activations import ACTS

NAMES = {
    "relu": "ReLU", "gelu": "GELU", "silu": "SiLU", "relu2": "ReLU$^2$", "swiglu": "SwiGLU",
    "groupgate": "Group Gate", "rankrelu2": "Rank-ReLU$^2$", "smoothrelu2": "Smooth-back ReLU$^2$",
    "crosssquare": "Cross-Square", "crosssquare_untied": "Cross-Square (untied)",
}
BASELINES = {"relu", "gelu", "silu", "relu2", "swiglu"}


def load(out: Path):
    runs = defaultdict(list)
    for p in out.glob("*.json"):
        r = json.loads(p.read_text())
        if not r.get("done"):
            continue
        c = r["config"]
        runs[(c["size"], c["act"], c["tag"], c["lr"])].append(r)
    return runs


def summarise(runs, size: str):
    rows = []
    for act in ACTS:
        by_lr = {lr: rs for (s, a, t, lr), rs in runs.items() if s == size and a == act and t == ""}
        seed0 = {lr: [r for r in rs if r["config"]["seed"] == 0] for lr, rs in by_lr.items()}
        seed0 = {lr: rs[0] for lr, rs in seed0.items() if rs}
        if not seed0:
            continue
        lr = min(seed0, key=lambda k: seed0[k]["best_val"])
        vals = [r["best_val"] for r in by_lr[lr]]
        tps = np.mean([r["tok_per_s"] for r in by_lr[lr]])
        rows.append(dict(act=act, lr=lr, n=len(vals), mean=np.mean(vals),
                         std=np.std(vals, ddof=1) if len(vals) > 1 else 0.0, tps=tps, curve=seed0[lr]["history"]))
    return rows


def write_tables(rows, size: str, res: Path):
    best = min(r["mean"] for r in rows)
    relu_tps = next((r["tps"] for r in rows if r["act"] == "relu"), None)
    md = [f"| activation | best lr | seeds | best val loss | tok/s vs ReLU |", "|---|---|---|---|---|"]
    tex = []
    for r in rows:
        speed = f"{r['tps'] / relu_tps:.2f}x" if relu_tps else "n/a"
        md.append(f"| {r['act']} | {r['lr']:g} | {r['n']} | {r['mean']:.4f} ± {r['std']:.4f} | {speed} |")
        cell = f"{r['mean']:.3f} $\\pm$ {r['std']:.3f}"
        if r["mean"] == best:
            cell = f"\\textbf{{{cell}}}"
        sep = "\\midrule\n" if r["act"] == "groupgate" else ""
        tex.append(f"{sep}{NAMES[r['act']]} & {cell} \\\\")
    (res / f"table_{size}.md").write_text("\n".join(md) + "\n")
    (res / f"table_{size}_rows.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


def plot(rows, size: str, res: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        h = [p for p in r["curve"] if p["step"] > 0]
        style = "--" if r["act"] in BASELINES else "-"
        ax.plot([p["tokens"] / 1e6 for p in h], [p["val"] for p in h], style, label=NAMES[r["act"]], lw=1.4)
    lo = min(r["mean"] for r in rows)
    ax.set_ylim(lo - 0.05, lo + 0.6)
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation loss")
    ax.set_title(f"Validation loss, {size} (seed 0, best learning rate)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(res / f"curves_{size}.pdf")
    fig.savefig(res / f"curves_{size}.png", dpi=150)


def ablations(runs, size: str, res: Path):
    lines = ["| run | lr | best val loss |", "|---|---|---|"]
    for (s, a, t, lr), rs in sorted(runs.items()):
        if s == size and (t or a == "crosssquare_untied"):
            lines.append(f"| {a}{t} | {lr:g} | {rs[0]['best_val']:.4f} |")
    (res / f"ablations_{size}.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="s20")
    p.add_argument("--out", default="runs")
    p.add_argument("--results", default="results")
    a = p.parse_args()
    res = Path(a.results)
    res.mkdir(exist_ok=True)
    runs = load(Path(a.out))
    rows = summarise(runs, a.size)
    if not rows:
        raise SystemExit(f"no finished runs for {a.size} in {a.out}")
    write_tables(rows, a.size, res)
    plot(rows, a.size, res)
    ablations(runs, a.size, res)
    print(f"wrote tables and plots to {res}/")


if __name__ == "__main__":
    main()
