"""CLI entry point.

Example:
  python main.py --task mnli --method tad --p 0.1 --T 3 --seed 0 \
    --output_dir runs/mnli_tad_p0.1_T3_s0
"""
from __future__ import annotations

import argparse
import os

from src.trainer import TrainConfig, train


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    ap = argparse.ArgumentParser()
    for k, v in cfg.__dict__.items():
        t = type(v)
        if t is bool:
            ap.add_argument(f"--{k}", action="store_true" if not v else "store_false")
        else:
            ap.add_argument(f"--{k}", type=t, default=v)
    args = ap.parse_args()
    for k in cfg.__dict__:
        setattr(cfg, k, getattr(args, k))
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    train(cfg)
