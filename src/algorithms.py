"""Per-method scheduling and mixing logic.

Each algorithm answers two questions every round t:
  - which factor is trained locally? ('A', 'B', or 'both')
  - which factors are mixed across clients? (subset of {'A','B'})

LoRA (vanilla DFL): both factors trained, both mixed (FedAvg-style P2P).
FFA-LoRA:           A frozen at init for everyone, only B trained and mixed.
RoLoRA:             alternating per round; only the active factor is mixed.
TAD-LoRA (ours):    alternating with interval T; BOTH factors mixed every round.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RoundPlan:
    trainable: str   # 'A' | 'B' | 'both'
    mix: Tuple[str, ...]  # subset of ('A','B')


def plan_round(method: str, t: int, T: int) -> RoundPlan:
    if method == "lora":
        return RoundPlan(trainable="both", mix=("A", "B"))
    if method == "ffa":
        # FFA-LoRA: A is frozen forever; only B trains and mixes
        return RoundPlan(trainable="B", mix=("B",))
    if method == "rolora":
        # Naive alternating extended to DFL: mix only the active block.
        # Paper uses T=1 for the RoLoRA baseline.
        phase_B = (t // max(T, 1)) % 2 == 0
        trainable = "B" if phase_B else "A"
        return RoundPlan(trainable=trainable, mix=(trainable,))
    if method == "tad":
        phase_B = (t // max(T, 1)) % 2 == 0
        trainable = "B" if phase_B else "A"
        return RoundPlan(trainable=trainable, mix=("A", "B"))
    raise ValueError(method)
