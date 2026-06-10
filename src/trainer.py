"""Decentralized FL trainer with multi-GPU client parallelism.

Architecture
------------
- The orchestrator (main process) holds the per-client LoRA state dicts in CPU
  RAM. With r=8 on RoBERTa-Large Q/V (24 layers), each client is < 2 MB.
- We spawn one worker process per GPU. Each worker loads ONE copy of
  RoBERTa-Large + LoRA on its assigned GPU, then services client jobs from a
  task queue. The base model never moves; only the LoRA factors are swapped.
- Each round: dispatch all m client local-update jobs across the workers,
  gather updated LoRA states, perform mixing on CPU (cheap), repeat.

This is a simulation, not real decentralized comms: every client's state lives
in the orchestrator. The mixing matrix W^t models the doubly-stochastic
peer-to-peer averaging that the paper analyzes.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.optim import AdamW

from .algorithms import RoundPlan, plan_round
from .client import evaluate, local_update
from .data import TASK_INFO, load_glue, partition_indices, tokenize
from .lora import LoRAConfig, get_lora_state, set_lora_state
from .model import build_model, build_tokenizer
from .topology import make_mixing


# ---------- Worker side ----------

def _worker_main(rank: int, world_size: int, task: str, lora_cfg: LoRAConfig,
                 task_q: mp.Queue, result_q: mp.Queue,
                 tokenized_path: str, val_path: str,
                 batch_size: int, eval_batch_size: int) -> None:
    """One persistent worker per GPU."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Load the (tokenized) datasets from disk to avoid re-tokenizing per worker.
    from datasets import load_from_disk
    train_ds = load_from_disk(tokenized_path)
    val_ds = load_from_disk(val_path)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    num_labels = TASK_INFO[task]["num_labels"]
    model = build_model(num_labels, lora_cfg).to(device)

    while True:
        msg = task_q.get()
        if msg is None:
            return
        kind = msg["kind"]

        if kind == "local":
            client_id = msg["client_id"]
            state = msg["state"]                # CPU dict
            indices = msg["indices"]            # np.ndarray
            trainable = msg["trainable"]
            local_steps = msg["local_steps"]
            lr = msg["lr"]
            set_lora_state(model, state)
            from .lora import set_factor_trainable
            set_factor_trainable(model, trainable)
            # fresh optimizer per round (HF AdamW defaults: wd=0, betas=(0.9,0.999), eps=1e-8)
            opt = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
            loss = local_update(model, opt, train_ds, indices,
                                trainable=trainable, local_steps=local_steps,
                                batch_size=batch_size, device=device)
            new_state = get_lora_state(model)
            result_q.put({"client_id": client_id, "state": new_state, "loss": loss})

        elif kind == "eval":
            client_id = msg["client_id"]
            state = msg["state"]
            set_lora_state(model, state)
            acc = evaluate(model, val_ds, eval_batch_size, device)
            result_q.put({"client_id": client_id, "acc": acc})


# ---------- Orchestrator ----------

@dataclass
class TrainConfig:
    task: str = "sst2"
    method: str = "tad"             # lora | ffa | rolora | tad
    num_clients: int = 100
    rounds: int = 150
    local_steps: int = 20
    batch_size: int = 32
    eval_batch_size: int = 64
    max_len: int = 128
    lr: float = 5e-4
    T: int = 1                       # switching interval
    topology: str = "er"             # er | ring
    p: float = 0.1                   # ER edge prob
    seed: int = 0
    eval_every: int = 25
    eval_clients: int = 10           # subsample clients for eval to save time
    output_dir: str = "runs/exp"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    world_size: int = 8

    def lora_cfg(self) -> LoRAConfig:
        return LoRAConfig(r=self.lora_r, alpha=self.lora_alpha,
                          dropout=self.lora_dropout)


def _stack_factor(states: List[Dict[str, torch.Tensor]], factor: str) -> Dict[str, torch.Tensor]:
    """Stack [m] client states into per-key tensors of shape [m, ...]."""
    keys = [k for k in states[0].keys() if k.endswith(f".{factor}")]
    return {k: torch.stack([s[k] for s in states], dim=0) for k in keys}


def _mix_factor(stacked: Dict[str, torch.Tensor], W: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Apply W [m,m] to a stacked tensor [m, ...] along dim 0."""
    out = {}
    for k, t in stacked.items():
        flat = t.reshape(t.shape[0], -1)            # [m, d]
        mixed = W @ flat                            # [m, d]
        out[k] = mixed.reshape(t.shape)
    return out


def _unstack_into(states: List[Dict[str, torch.Tensor]], mixed: Dict[str, torch.Tensor],
                  factor: str) -> None:
    for k, t in mixed.items():
        for i, s in enumerate(states):
            s[k] = t[i].clone()


def _prepare_data(cfg: TrainConfig, tokenizer, cache_root: str) -> Tuple[str, str, List[np.ndarray]]:
    """Tokenize once, persist to disk, return paths + client index partition."""
    os.makedirs(cache_root, exist_ok=True)
    train_path = os.path.join(cache_root, f"{cfg.task}_train_l{cfg.max_len}")
    val_path = os.path.join(cache_root, f"{cfg.task}_val_l{cfg.max_len}")
    if not os.path.isdir(train_path) or not os.path.isdir(val_path):
        train_raw, val_raw = load_glue(cfg.task)
        train_tok = tokenize(train_raw, cfg.task, tokenizer, cfg.max_len)
        val_tok = tokenize(val_raw, cfg.task, tokenizer, cfg.max_len)
        train_tok.save_to_disk(train_path)
        val_tok.save_to_disk(val_path)
        labels = np.array(train_raw["label"])
    else:
        from datasets import load_from_disk
        train_raw, _ = load_glue(cfg.task)
        labels = np.array(train_raw["label"])

    parts = partition_indices(
        labels, cfg.num_clients, TASK_INFO[cfg.task]["num_labels"], seed=cfg.seed
    )
    return train_path, val_path, parts


def _init_client_states(cfg: TrainConfig) -> List[Dict[str, torch.Tensor]]:
    """Build one CPU model briefly to capture initial LoRA state, then replicate."""
    num_labels = TASK_INFO[cfg.task]["num_labels"]
    tmp = build_model(num_labels, cfg.lora_cfg())
    init = get_lora_state(tmp)
    del tmp
    # All clients start identical (paper assumption).
    return [{k: v.clone() for k, v in init.items()} for _ in range(cfg.num_clients)]


def train(cfg: TrainConfig) -> Dict[str, Any]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = build_tokenizer()
    cache_root = os.environ.get("FL_DATA_CACHE", "data_cache")
    train_path, val_path, parts = _prepare_data(cfg, tokenizer, cache_root)

    states = _init_client_states(cfg)

    # spin up workers
    mp.set_start_method("spawn", force=True)
    task_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()
    procs = []
    world = min(cfg.world_size, torch.cuda.device_count())
    if world == 0:
        raise RuntimeError("No CUDA devices available")
    for r in range(world):
        p = mp.Process(target=_worker_main,
                       args=(r, world, cfg.task, cfg.lora_cfg(),
                             task_q, result_q, train_path, val_path,
                             cfg.batch_size, cfg.eval_batch_size))
        p.start()
        procs.append(p)

    log: List[Dict[str, Any]] = []
    try:
        for t in range(cfg.rounds):
            plan = plan_round(cfg.method, t, cfg.T)
            # dispatch all client local updates
            for i in range(cfg.num_clients):
                task_q.put({
                    "kind": "local",
                    "client_id": i,
                    "state": states[i],
                    "indices": parts[i],
                    "trainable": plan.trainable,
                    "local_steps": cfg.local_steps,
                    "lr": cfg.lr,
                })
            losses = [0.0] * cfg.num_clients
            done = 0
            while done < cfg.num_clients:
                res = result_q.get()
                cid = res["client_id"]
                states[cid] = res["state"]
                losses[cid] = res["loss"]
                done += 1

            # mixing
            W_np = make_mixing(cfg.topology, cfg.num_clients,
                               cfg.p if cfg.topology == "er" else None, rng)
            W = torch.tensor(W_np, dtype=torch.float32)
            if "A" in plan.mix:
                stacked = _stack_factor(states, "A")
                mixed = _mix_factor(stacked, W)
                _unstack_into(states, mixed, "A")
            if "B" in plan.mix:
                stacked = _stack_factor(states, "B")
                mixed = _mix_factor(stacked, W)
                _unstack_into(states, mixed, "B")

            entry = {"round": t, "mean_loss": float(np.mean(losses)),
                     "trainable": plan.trainable, "mix": list(plan.mix)}

            if (t + 1) % cfg.eval_every == 0 or t == cfg.rounds - 1:
                eval_ids = rng.choice(cfg.num_clients,
                                      size=min(cfg.eval_clients, cfg.num_clients),
                                      replace=False)
                for cid in eval_ids:
                    task_q.put({"kind": "eval", "client_id": int(cid),
                                "state": states[int(cid)]})
                accs = []
                done = 0
                while done < len(eval_ids):
                    res = result_q.get()
                    accs.append(res["acc"])
                    done += 1
                entry["mean_acc"] = float(np.mean(accs))
                entry["std_acc"] = float(np.std(accs))
                print(f"[{cfg.task}/{cfg.method}/p={cfg.p}/T={cfg.T}/seed={cfg.seed}] "
                      f"round {t+1}/{cfg.rounds} loss={entry['mean_loss']:.4f} "
                      f"acc={entry['mean_acc']:.4f}", flush=True)

            log.append(entry)

        # final eval over ALL clients (paper protocol)
        for cid in range(cfg.num_clients):
            task_q.put({"kind": "eval", "client_id": cid, "state": states[cid]})
        accs = [0.0] * cfg.num_clients
        done = 0
        while done < cfg.num_clients:
            res = result_q.get()
            accs[res["client_id"]] = res["acc"]
            done += 1
        final_acc = float(np.mean(accs))
        final_std = float(np.std(accs))
        print(f"[FINAL {cfg.task}/{cfg.method}/p={cfg.p}/T={cfg.T}/seed={cfg.seed}] "
              f"mean_acc={final_acc:.4f} +/- {final_std:.4f}", flush=True)

    finally:
        for _ in procs:
            task_q.put(None)
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    result = {
        "config": cfg.__dict__,
        "log": log,
        "final_mean_acc": final_acc,
        "final_std_acc": final_std,
        "per_client_acc": accs,
    }
    import json
    with open(os.path.join(cfg.output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
