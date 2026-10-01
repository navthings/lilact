import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

TOKENIZER = "hf-internal-testing/llama-tokenizer"
EOS = 2


def write_split(ds, tok, path: Path, max_tokens: int | None, batch: int = 2000):
    # stream stories in batches and append uint16 ids, so the full set never sits in memory
    n = 0
    with open(path, "wb") as f:
        for i in range(0, len(ds), batch):
            texts = ds[i : i + batch]["text"]
            ids = []
            for enc in tok.encode_batch(texts, add_special_tokens=False):
                ids.extend(enc.ids)
                ids.append(EOS)
            arr = np.asarray(ids, dtype=np.uint16)
            if max_tokens is not None and n + len(arr) > max_tokens:
                arr = arr[: max_tokens - n]
            f.write(arr.tobytes())
            n += len(arr)
            if max_tokens is not None and n >= max_tokens:
                break
            if (i // batch) % 50 == 0:
                print(f"{path.name}: {n / 1e6:.1f}M tokens", flush=True)
    print(f"{path.name}: done, {n:,} tokens")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tinystories")
    p.add_argument("--max_train_tokens", type=int, default=None)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_pretrained(TOKENIZER)
    if tok.get_vocab_size() > 65535:
        raise ValueError("vocab too big for uint16")

    ds = load_dataset("roneneldan/TinyStories")
    write_split(ds["validation"], tok, out / "val.bin", None)
    write_split(ds["train"], tok, out / "train.bin", args.max_train_tokens)
    (out / "vocab_size.txt").write_text(str(tok.get_vocab_size()))


if __name__ == "__main__":
    main()
