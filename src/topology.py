"""Time-varying mixing matrices for decentralized FL.

We use the Erdős–Rényi edge-activation gossip model from the paper:
  - underlying graph G_0 = complete graph on m nodes
  - at each round, every edge is independently active with probability p
  - the mixing matrix is built by Metropolis–Hastings weights on the active
    edges, which produces a doubly-stochastic matrix with the spectral-gap
    scaling derived in Appendix A.J.

A ring topology is also provided for stress-test runs (Appendix D).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def metropolis_hastings(adj: np.ndarray) -> np.ndarray:
    """Symmetric doubly-stochastic mixing from a 0/1 adjacency (no self-loops)."""
    m = adj.shape[0]
    deg = adj.sum(axis=1)
    W = np.zeros_like(adj, dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i != j and adj[i, j]:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
    for i in range(m):
        W[i, i] = 1.0 - W[i].sum()
    return W


def erdos_renyi_mixing(m: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Sample an ER(p) adjacency on m nodes, return MH mixing matrix."""
    iu = np.triu_indices(m, k=1)
    mask = rng.random(len(iu[0])) < p
    adj = np.zeros((m, m), dtype=np.int8)
    adj[iu[0][mask], iu[1][mask]] = 1
    adj = adj + adj.T
    return metropolis_hastings(adj)


def ring_mixing(m: int) -> np.ndarray:
    adj = np.zeros((m, m), dtype=np.int8)
    for i in range(m):
        adj[i, (i + 1) % m] = 1
        adj[i, (i - 1) % m] = 1
    return metropolis_hastings(adj)


def make_mixing(kind: str, m: int, p: Optional[float], rng: np.random.Generator) -> np.ndarray:
    if kind == "er":
        assert p is not None
        return erdos_renyi_mixing(m, p, rng)
    if kind == "ring":
        return ring_mixing(m)
    raise ValueError(kind)
