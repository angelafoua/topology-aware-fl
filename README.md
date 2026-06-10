# TAD-LoRA Reproduction

Reproduction of *"Stabilizing Decentralized Federated Fine-Tuning via Topology-Aware
Alternating LoRA"* (Wang et al., 2026) — scaled to **100 clients** on **8x NVIDIA A100**.

## What is implemented

- RoBERTa-Large + custom LoRA (r=8, alpha=16, dropout=0.1) on Q/V projections, frozen
  classification head — matches the paper's model config (Sec VI.A.1).
- 4 GLUE tasks: SST-2, QNLI, QQP, MNLI.
- 100 non-IID clients. The paper's 10-client mixture
  (binary: `3x[0.9,0.1], 3x[0.1,0.9], 4x[0.5,0.5]`; MNLI: `4x[0.9,0.05,0.05], 3x[0.05,0.9,0.05], 3x[0.05,0.05,0.9]`)
  is replicated 10x to scale to 100 clients while preserving the skew distribution.
- Erdos-Renyi communication topology with edge-activation probability
  p in {0.5, 0.2, 0.1, 0.05, 0.02, 0.01}, plus a ring topology for the appendix
  stress test. Doubly-stochastic mixing via Metropolis-Hastings weights.
- Four methods (`src/algorithms.py`):
  - `lora`   - vanilla decentralized LoRA: both factors trained and mixed.
  - `ffa`    - FFA-LoRA: A frozen at init, only B trains and mixes.
  - `rolora` - RoLoRA: per-round alternating, only the active factor is mixed.
  - `tad`    - TAD-LoRA: interval-T alternating, both factors mixed each round.
- 150 rounds x 20 local steps x batch 32 x seq-len 128, AdamW (HF defaults).
- LR grid: {2e-4, 5e-4, 1e-3, 2e-3, 5e-3}. Switching intervals T in {1,2,3,5,10,15}.

## Layout

```
src/
  data.py        # GLUE loading + 100-client non-IID partition
  lora.py        # Custom LoRA injection (so we can freeze A or B independently)
  model.py       # RoBERTa-Large + LoRA builder
  topology.py    # ER and ring mixing matrices (Metropolis-Hastings)
  algorithms.py  # Per-method round plan (trainable factor + mixed factors)
  client.py      # Local SGD step + evaluation
  trainer.py     # Multi-GPU client-parallel orchestrator
main.py          # CLI entry point
scripts/
  run_single.sh    # One (task,method,p,T,lr,seed) run
  run_lr_search.sh # Paper LR grid at p=0.5
  run_grid.sh      # Full headline grid
  aggregate.py     # Collect runs/*/result.json -> summary.csv
```

## Multi-GPU parallelism

Each round, 100 clients run local SGD independently. The trainer spawns one
persistent worker process per GPU (8 total); each worker keeps one copy of
RoBERTa-Large resident on its device, swapping only the small LoRA factors per
client. Mixing W in R^(100x100) is applied on CPU between rounds.

## How to run on 8x A100

```bash
pip install -r requirements.txt

# 1. (Optional) per-task LR search at p=0.5 to pick LR_DEFAULT.
bash scripts/run_lr_search.sh

# 2. Headline grid (Table I subset, p in {0.5,0.1,0.02}).
PS="0.5 0.1 0.02" SEEDS="0 1 2" bash scripts/run_grid.sh

# 3. Full grid + T sweep for TAD-LoRA (Figures 3, 4).
bash scripts/run_grid.sh

# 4. Aggregate.
python scripts/aggregate.py
```

Override the defaults via env vars: `TASKS`, `METHODS`, `PS`, `TS`, `SEEDS`,
`LR_DEFAULT`. Example:

```bash
TASKS="mnli" METHODS="tad" PS="0.02" TS="1 3 5 10 15" bash scripts/run_grid.sh
```

## Compute budget (rough estimate)

A single (task, method, p, T, seed) run is 100 clients x 20 steps x 150 rounds =
300k local SGD steps on RoBERTa-Large. Distributed across 8 A100s, expect
roughly 1-2 hours per run depending on task (MNLI longest). The full paper grid
(4 tasks x 6 p x ~12 method-T combos x 3 seeds ~= 850 runs) is ~1500 GPU-hours.
Use the `PS`/`TS` env vars to focus on the cells you care about.

## Eval protocol

Following the paper: at the end of training, evaluate every client's model on
the task's validation split and report mean +/- std across clients. Intermediate
evals use a random subsample of 10 clients (configurable via `--eval_clients`).

## Deviations from the paper

- 100 clients vs. 10 in the paper. The label-skew mixture is preserved by
  replicating each bucket 10x.
- We use Metropolis-Hastings mixing weights on the ER-sampled adjacency, which
  matches the doubly-stochastic / spectral-gap-scaling assumption used in the
  paper's Appendix A.J but is one specific instantiation; the paper does not
  pin down a particular weighting rule.
