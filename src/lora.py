"""Minimal LoRA injection for RoBERTa Q/V projections.

We implement LoRA from scratch (rather than using PEFT) so we have direct
control over the A/B factors and can:
  - freeze A while training B  (FFA-LoRA)
  - swap which factor is active each round  (RoLoRA, TAD-LoRA)
  - mix the A and B factors independently across clients
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("query", "value")


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with trainable A (r x in) and B (out x r)."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scaling


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> List[str]:
    """Replace every nn.Linear whose name ends in a target module with LoRALinear.
    Returns the list of LoRA module names (parent paths).
    """
    injected: List[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in cfg.target_modules:
                continue
            wrapped = LoRALinear(child, cfg.r, cfg.alpha, cfg.dropout)
            setattr(module, child_name, wrapped)
            injected.append(f"{name}.{child_name}" if name else child_name)
    return injected


def lora_modules(model: nn.Module) -> Iterable[Tuple[str, LoRALinear]]:
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            yield n, m


def get_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a CPU state dict of all LoRA A/B factors."""
    state: Dict[str, torch.Tensor] = {}
    for n, m in lora_modules(model):
        state[f"{n}.A"] = m.lora_A.detach().cpu().clone()
        state[f"{n}.B"] = m.lora_B.detach().cpu().clone()
    return state


def set_lora_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for n, m in lora_modules(model):
            m.lora_A.copy_(state[f"{n}.A"].to(m.lora_A.device))
            m.lora_B.copy_(state[f"{n}.B"].to(m.lora_B.device))


def set_factor_trainable(model: nn.Module, factor: str) -> None:
    """factor in {'A','B','both','none'} -- enables grad only on that factor."""
    for _, m in lora_modules(model):
        m.lora_A.requires_grad = factor in ("A", "both")
        m.lora_B.requires_grad = factor in ("B", "both")
