# lilact

Code and experiments for my paper *Group Gate, Cross-Square and Squared Surrogates: New Activations for Small Transformers*.

ReLU, GELU and SiLU are basically tied in published comparisons, and gated (SwiGLU) and squared (ReLU²) MLPs beat all three. This repo tests four new activations that combine those ideas, against those five baselines, in small Llama-style transformers trained in MLX on a MacBook.

## The activations

All of them live in `activations.py`. `h` is the MLP's up-projection output.

| name | flag | what it does |
|---|---|---|
| ReLU | `relu` | `max(0, h)` |
| GELU | `gelu` | `h * Φ(h)` |
| SiLU | `silu` | `h * sigmoid(h)` |
| ReLU² | `relu2` | `max(0, h)²` |
| SwiGLU | `swiglu` | `silu(W x) * (V x)`, hidden width cut to 8/3 d to match params |
| **Group Gate** | `groupgate` | each unit is gated by a learned mix of the `g` units in its group; starts as exactly SiLU |
| **Rank-ReLU²** | `rankrelu2` | ReLU² with a per-token cutoff at `mean + τ·std`, divided by std; `τ` learned per layer |
| **Smooth-back ReLU²** | `smoothrelu2` | ReLU² forward, smooth surrogate gradient backward so negative units still learn |
| **Cross-Square** | `crosssquare` | split `h` into halves `u, v`; output `[relu(u)² * sigmoid(v), relu(v)² * sigmoid(u)]` |

Every model has the same parameter count within about 1% (Group Gate adds `d_ff * g` per layer).

## Setup

```
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python test_activations.py
python prepare.py
```

`prepare.py` downloads TinyStories and tokenises it with the Llama tokenizer (32k vocab) into `data/tinystories/*.bin`, streamed in batches so it never holds the whole set in memory.

## Running

One run:

```
python train.py --act crosssquare --size s20 --lr 1e-3 --seed 0
```

The full grid, in three stages. Each stage skips runs that already finished, so you can stop and restart it.

```
python sweep.py --stage lr          # every activation at lr 5e-4, 1e-3, 2e-3, seed 0
python sweep.py --stage seeds       # best lr per activation, seeds 1 and 2
python sweep.py --stage ablations   # g=2/32, stop-grad, beta=1/4, untied cross-square
python analyze.py --size s20        # tables + loss curves into results/
```

`analyze.py` writes `results/table_s20_rows.tex`, which pastes straight into the paper's main table.

## Sizes and cost

| size | d | layers | heads | params | default tokens | tokens/step |
|---|---|---|---|---|---|---|
| `s20` | 384 | 6 | 6 | ~23M | 100M | 16k |
| `s100` | 768 | 12 | 12 | ~110M | 300M | 32k |

Each JSON in `runs/` stores the full val-loss history, best and final loss, tokens/s, the share of dead and zero units per layer, the largest activation per layer, and the learned parameters (Group Gate's `A`, Rank-ReLU²'s `τ`).

If training is slow and `sysctl vm.swapusage` shows swap filling up, the 32k-vocab logits do not fit next to your open apps: halve `--micro_bs` and double `--accum` (same tokens per step).

On a fanless MacBook Air the full `s20` grid (45 runs) is days of compute, and the chip throttles once it heats up, so tokens/s numbers are only comparable between runs started cold. `s100` is better on cloud hardware.

## Stage 2: scaling up on a TPU

`tpu/scaleup_tpu_kaggle.ipynb` trains the stage-1 winner against SwiGLU at ~520M params (sprout's config: d=1280, 28 layers, 20/4 heads) on 2.5B FineWeb-Edu tokens, on a Kaggle TPU v5e-8. It's sprout's training code with the activation swapped in, so it keeps ZeRO-1 sharding, resuming across sessions and the HBM-overflow fallback.

Make two copies on Kaggle, set `ACT` in the config cell (`"swiglu"` in one, the winner in the other), and run both. Each takes ~5-6h, so one session each, about 11h of the 20h weekly TPU quota. The output has `result_<act>.json` with the numbers for the paper's stage-2 table.

`tpu/scaleup.py` is the same notebook as a plain script.

## Layout

```
activations.py       the nine activations + MLP block
model.py             Llama-style transformer (RMSNorm, RoPE, tied embeddings, no biases)
data.py              shuffled non-overlapping windows, no token seen twice
prepare.py           TinyStories -> uint16 token files
train.py             one run -> runs/<name>.json
sweep.py             lr sweep, seeds, ablations
analyze.py           tables and plots
test_activations.py  worked examples, gradient checks, param counts
pilot/               first char-level pilot on Tiny Shakespeare
tpu/                 stage-2 kaggle notebook (jax, tpu v5e-8)
paper/               LaTeX source (NeurIPS 2026 template, preprint mode)
```
