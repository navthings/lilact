from pathlib import Path

import mlx.core as mx
import numpy as np


class Batches:
    """Non-overlapping windows in a shuffled order, so no token is seen twice until the set runs out."""

    def __init__(self, path: Path, seq: int, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq = seq
        n_windows = (len(self.data) - 1) // seq
        self.order = np.random.default_rng(seed).permutation(n_windows)
        self.pos = 0

    def tokens_available(self) -> int:
        return len(self.order) * self.seq

    def next(self, batch: int):
        if self.pos + batch > len(self.order):
            raise RuntimeError("ran out of unseen training data, lower --tokens or prepare more data")
        idx = self.order[self.pos : self.pos + batch]
        self.pos += batch
        x = np.stack([self.data[i * self.seq : i * self.seq + self.seq + 1] for i in idx]).astype(np.int32)
        return mx.array(x[:, :-1]), mx.array(x[:, 1:])


def val_batches(path: Path, seq: int, batch: int, n_batches: int):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    need = n_batches * batch * seq + 1
    if len(data) < need:
        raise RuntimeError(f"val set has {len(data):,} tokens, need {need:,}")
    out = []
    for b in range(n_batches):
        start = b * batch * seq
        x = np.stack([data[start + i * seq : start + i * seq + seq + 1] for i in range(batch)]).astype(np.int32)
        out.append((mx.array(x[:, :-1]), mx.array(x[:, 1:])))
    return out
