"""Local client update and federated evaluation primitives."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from .lora import set_factor_trainable


def _trainable_params(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def local_update(model, optimizer, dataset, indices: np.ndarray, *,
                 trainable: str, local_steps: int, batch_size: int,
                 device: torch.device) -> float:
    """Run `local_steps` SGD steps on the given client subset.

    The caller is responsible for having loaded the client's (A,B) into `model`
    before calling. Returns mean loss.
    """
    set_factor_trainable(model, trainable)
    model.train()
    if len(indices) == 0:
        return 0.0
    rng = np.random.RandomState()  # nondeterministic per call; seed handled outside
    sub = Subset(dataset, indices.tolist())
    loader = DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=0,
                        drop_last=False, pin_memory=True)
    loss_sum, n = 0.0, 0
    it = iter(loader)
    for _ in range(local_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_trainable_params(model), 1.0)
        optimizer.step()
        loss_sum += loss.item()
        n += 1
    return loss_sum / max(n, 1)


@torch.no_grad()
def evaluate(model, dataset, batch_size: int, device: torch.device) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct, total = 0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(**batch).logits
        pred = logits.argmax(dim=-1)
        correct += (pred == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    return correct / max(total, 1)
