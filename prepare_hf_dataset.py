"""
Prepares azure_qa_pairs.jsonl for training and pushes it to the HF Hub as a
proper dataset repo (train/validation/test splits, with a dataset card).

Run this BEFORE train.ipynb — the notebook expects a pushed (or locally
saved) DatasetDict, not the raw jsonl.
"""

import json
import hashlib
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi

QA_FILE = Path("./azure_qa_pairs.jsonl")
HF_DATASET_REPO = "your-username/azure-docs-qa"   # <-- change this
PUSH_PUBLIC = False                                # <-- flip deliberately, see note above
LOCAL_SAVE_DIR = Path("./azure_qa_dataset")


def load_pairs():
    pairs = []
    with QA_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"[prepare] loaded {len(pairs)} raw pairs")
    return pairs


def dedupe(pairs):
    """
    Crude but effective: normalize whitespace/case on the question, hash it,
    drop repeats. Azure docs have near-duplicate content across regional/
    language variants — without this your eval split gets contaminated by
    near-duplicates of training examples and your eval numbers lie to you.
    """
    seen = set()
    deduped = []
    for p in pairs:
        key = hashlib.md5(p["instruction"].strip().lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    print(f"[prepare] {len(pairs)} -> {len(deduped)} after dedupe")
    return deduped


def basic_quality_filter(pairs):
    """Drop pairs that are obviously broken — empty fields, too-short answers."""
    filtered = []
    for p in pairs:
        q, a = p.get("instruction", "").strip(), p.get("response", "").strip()
        if len(q) < 10 or len(a) < 20:
            continue
        filtered.append(p)
    print(f"[prepare] {len(pairs)} -> {len(filtered)} after quality filter")
    return filtered


def build_splits(pairs, seed=42):
    ds = Dataset.from_list(pairs)
    ds = ds.shuffle(seed=seed)

    n = len(ds)
    train_end = int(n * 0.85)
    val_end = int(n * 0.95)

    splits = DatasetDict({
        "train": ds.select(range(0, train_end)),
        "validation": ds.select(range(train_end, val_end)),
        "test": ds.select(range(val_end, n)),
    })
    for name, split in splits.items():
        print(f"[prepare] {name}: {len(split)} examples")
    return splits


DATASET_CARD = """---
license: cc-by-4.0
language:
- en
tags:
- azure
- question-answering
- instruction-tuning
---

# Azure Docs QA Dataset

Instruction/response pairs generated from Microsoft's official Azure
documentation (`MicrosoftDocs/azure-docs`, CC BY 4.0), for fine-tuning
a small model to answer Azure-suite questions.

**Generation method:** each doc chunk was passed to an LLM with instructions
to produce grounded Q/A pairs answerable strictly from that chunk. Not
manually verified beyond a spot-check sample — treat as noisy supervised
data, not a gold-standard benchmark.

**Source attribution:** derived from Microsoft Learn / azure-docs,
licensed CC BY 4.0. https://github.com/MicrosoftDocs/azure-docs

**Splits:** train / validation / test, deduplicated on question text,
random 85/10/5 split, seed=42.
"""


def push_to_hub(splits: DatasetDict):
    login()  # prompts for HF token if not already cached
    splits.push_to_hub(HF_DATASET_REPO, private=not PUSH_PUBLIC)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"[prepare] pushed to https://huggingface.co/datasets/{HF_DATASET_REPO}")


if __name__ == "__main__":
    pairs = load_pairs()
    pairs = basic_quality_filter(pairs)
    pairs = dedupe(pairs)
    splits = build_splits(pairs)

    splits.save_to_disk(str(LOCAL_SAVE_DIR))
    print(f"[prepare] also saved locally to {LOCAL_SAVE_DIR}")

    # Comment this out if you just want the local copy for now.
    push_to_hub(splits)
