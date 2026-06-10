"""GLUE loading and non-IID partitioning across N clients.

We follow the paper's partition recipe and scale it to N clients by replicating
the per-block proportions. For 100 clients we use the same skew distribution
repeated 10x, so each label-skew bucket has 10 clients instead of the paper's 3-4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


TASK_INFO: Dict[str, Dict] = {
    "sst2":  {"keys": ("sentence", None),        "num_labels": 2, "metric_split": "validation"},
    "qnli":  {"keys": ("question", "sentence"),  "num_labels": 2, "metric_split": "validation"},
    "qqp":   {"keys": ("question1", "question2"),"num_labels": 2, "metric_split": "validation"},
    "mnli":  {"keys": ("premise", "hypothesis"), "num_labels": 3, "metric_split": "validation_matched"},
}


def load_glue(task: str) -> Tuple[Dataset, Dataset]:
    ds = load_dataset("nyu-mll/glue", task)
    train = ds["train"]
    val_split = TASK_INFO[task]["metric_split"]
    val = ds[val_split]
    return train, val


def tokenize(ds: Dataset, task: str, tok: PreTrainedTokenizerBase, max_len: int) -> Dataset:
    k1, k2 = TASK_INFO[task]["keys"]

    def fn(batch):
        if k2 is None:
            enc = tok(batch[k1], truncation=True, padding="max_length", max_length=max_len)
        else:
            enc = tok(batch[k1], batch[k2], truncation=True, padding="max_length", max_length=max_len)
        enc["labels"] = batch["label"]
        return enc

    cols = [c for c in ds.column_names if c not in ("idx",)]
    out = ds.map(fn, batched=True, remove_columns=cols)
    out.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return out


def _block_proportions(num_labels: int, num_clients: int) -> List[List[float]]:
    """Return a list of [num_clients] label-mixture vectors.

    Paper recipe (10 clients):
      binary tasks: 3 x [0.9,0.1], 3 x [0.1,0.9], 4 x [0.5,0.5]
      MNLI (3-way): 4 x [0.9,0.05,0.05], 3 x [0.05,0.9,0.05], 3 x [0.05,0.05,0.9]

    We scale to N clients by replicating the buckets proportionally.
    """
    rng = np.random.RandomState(0)
    if num_labels == 2:
        blocks = [([0.9, 0.1], 3), ([0.1, 0.9], 3), ([0.5, 0.5], 4)]
    elif num_labels == 3:
        blocks = [([0.9, 0.05, 0.05], 4), ([0.05, 0.9, 0.05], 3), ([0.05, 0.05, 0.9], 3)]
    else:
        raise ValueError(num_labels)
    base_total = sum(c for _, c in blocks)  # 10
    scale = num_clients // base_total
    rem = num_clients - scale * base_total
    counts = [c * scale for _, c in blocks]
    # distribute remainder round-robin over buckets
    for i in range(rem):
        counts[i % len(counts)] += 1
    out: List[List[float]] = []
    for (props, _), c in zip(blocks, counts):
        for _ in range(c):
            out.append(list(props))
    rng.shuffle(out)
    return out


def partition_indices(labels: np.ndarray, num_clients: int, num_labels: int,
                      seed: int = 0) -> List[np.ndarray]:
    """Allocate sample indices to clients matching the requested label mixture."""
    rng = np.random.RandomState(seed)
    props = _block_proportions(num_labels, num_clients)
    # per-label index pools, shuffled
    pools: Dict[int, np.ndarray] = {}
    for y in range(num_labels):
        idx = np.where(labels == y)[0]
        rng.shuffle(idx)
        pools[y] = idx
    cursors = {y: 0 for y in range(num_labels)}
    # samples per client: split evenly
    n_total = len(labels)
    per_client = n_total // num_clients
    client_idx: List[List[int]] = [[] for _ in range(num_clients)]
    for i, mix in enumerate(props):
        for y, frac in enumerate(mix):
            take = int(round(per_client * frac))
            avail = len(pools[y]) - cursors[y]
            take = min(take, avail)
            chunk = pools[y][cursors[y]:cursors[y] + take]
            cursors[y] += take
            client_idx[i].extend(chunk.tolist())
    # fallback: top up any starved client with random leftovers
    leftover = []
    for y in range(num_labels):
        leftover.extend(pools[y][cursors[y]:].tolist())
    rng.shuffle(leftover)
    li = 0
    for i in range(num_clients):
        deficit = per_client - len(client_idx[i])
        if deficit > 0:
            client_idx[i].extend(leftover[li:li + deficit])
            li += deficit
        rng.shuffle(client_idx[i])
    return [np.array(c, dtype=np.int64) for c in client_idx]
