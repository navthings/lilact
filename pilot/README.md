# pilot

First quick test before the real setup: a 4.8M-param character-level transformer (6 layers, d=256) on Tiny Shakespeare, 2000 steps, 2 seeds each, on a MacBook Air M5. `pilot.py` is the script, `results.jsonl` the raw runs.

It tested an earlier idea, `silusq` (SiLU plus a learned dose of its own square, `s + a*s²`), that was later dropped because it's in the same family as PolyCom and xIELU.

| activation | best val loss, seed 0 / 1 | mean |
|---|---|---|
| GELU | 1.4972 / 1.5053 | 1.5013 |
| SiLU² (plain MLP) | 1.5007 / 1.5052 | 1.5030 |
| ReLU² | 1.5061 / 1.5133 | 1.5097 |
| SiLU² (gated) | 1.5184 / 1.5200 | 1.5192 |
| SwiGLU | 1.5152 / 1.5251 | 1.5202 |

Lessons that shaped the real setup:

- Val loss bottoms out around step 1250 to 1750 and then rises: 4.8M params see the 1.1M-character set about 16 times, so the models memorise. Final loss here measures overfitting, not quality. The real runs use TinyStories with no repeated tokens.
- Gated MLPs came last, the opposite of the published ranking. Likely cause: gating is known to speed up optimisation, and in an overfitting regime faster fitting just means faster memorising. Not evidence against gating.
- The learned `a` in SiLU² grew with depth in both seeds (plain: about 0.02 in layer 1 to 0.11 in layer 6; gated: about -0.02 to 0.08), so the deeper layers wanted more of the squared term.
