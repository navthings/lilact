import argparse
import json
import subprocess
import sys
from pathlib import Path

from activations import ACTS

LRS = [5e-4, 1e-3, 2e-3]
SEEDS = [1, 2]
# (act, tag, extra args) for the ablations in the paper; main runs use g=8, beta=1.5
ABLATIONS = [
    ("groupgate", "_g2", ["--g", "2"]),
    ("groupgate", "_g32", ["--g", "32"]),
    ("rankrelu2", "_sg", ["--rank_stop_grad"]),
    ("smoothrelu2", "_b1", ["--beta", "1"]),
    ("smoothrelu2", "_b4", ["--beta", "4"]),
    ("crosssquare_untied", "", []),
]


def result(out: Path, size: str, act: str, tag: str, lr: float, seed: int):
    p = out / f"{size}_{act}{tag}_lr{lr:g}_s{seed}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r if r.get("done") else None


def best_lr(out: Path, size: str, act: str) -> float:
    found = {lr: result(out, size, act, "", lr, 0) for lr in LRS}
    missing = [lr for lr, r in found.items() if r is None]
    if missing:
        raise RuntimeError(f"{act}: lr sweep not finished, missing {missing}; run --stage lr first")
    return min(LRS, key=lambda lr: found[lr]["best_val"])


def jobs(stage: str, size: str, out: Path, acts: list[str]):
    if stage == "lr":
        return [(act, "", lr, 0, []) for act in acts for lr in LRS]
    if stage == "seeds":
        return [(act, "", best_lr(out, size, act), s, []) for act in acts for s in SEEDS]
    if stage == "ablations":
        base = {a: best_lr(out, size, a if a != "crosssquare_untied" else "crosssquare") for a, _, _ in ABLATIONS}
        return [(a, tag, base[a], 0, extra) for a, tag, extra in ABLATIONS]
    raise ValueError(stage)


def main():
    p = argparse.ArgumentParser(description="runs one stage of the experiment grid, skipping finished runs")
    p.add_argument("--stage", required=True, choices=["lr", "seeds", "ablations"])
    p.add_argument("--size", default="s20")
    p.add_argument("--acts", nargs="*", default=ACTS)
    p.add_argument("--out", default="runs")
    p.add_argument("--dry_run", action="store_true")
    a, passthrough = p.parse_known_args()
    out = Path(a.out)

    todo = [j for j in jobs(a.stage, a.size, out, a.acts) if result(out, a.size, j[0], j[1], j[2], j[3]) is None]
    print(f"{len(todo)} runs to do")
    for act, tag, lr, seed, extra in todo:
        cmd = [sys.executable, "train.py", "--act", act, "--size", a.size, "--lr", str(lr), "--seed", str(seed),
               "--out", str(out), "--tag", tag, *extra, *passthrough]
        print(" ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
