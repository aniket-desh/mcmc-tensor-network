"""Shared builders + CSV schema for dev-branch experiments.

These helpers mirror the network families used in the positive-only test suite
(2x2 ring, periodic MPS, 3x3 grid) so we can reuse the same instance generators
when adding signs/phases.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
import networkx as nx

from src.algorithm import TensorNetwork, contract_tensor_network


# ---- network builders -----------------------------------------------------

def build_2x2_ring(dim: int, seed: int) -> Tuple[nx.Graph, Dict]:
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
    rng = np.random.default_rng(seed)
    tensors = {}
    inds = {"A": ["i", "j"], "B": ["j", "k"], "C": ["k", "l"], "D": ["l", "i"]}
    for node, lab in inds.items():
        data = rng.normal(loc=1.0, scale=0.1, size=(dim, dim)) + 1e-3
        tensors[node] = (data, lab)
    return G, tensors


def build_periodic_mps(n_sites: int, dim: int, seed: int) -> Tuple[nx.Graph, Dict]:
    G = nx.Graph()
    rng = np.random.default_rng(seed)
    tensors = {}
    for s in range(n_sites):
        left = f"e{s}"
        right = f"e{(s + 1) % n_sites}"
        name = f"T{s}"
        G.add_edge(name, f"T{(s + 1) % n_sites}")
        data = rng.normal(loc=1.0, scale=0.1, size=(dim, dim)) + 1e-3
        tensors[name] = (data, [left, right])
    return G, tensors


def build_3x3_grid(dim: int, seed: int) -> Tuple[nx.Graph, Dict]:
    """Open-boundary 3x3 grid of rank-up-to-4 tensors with positive entries."""
    G = nx.Graph()
    rng = np.random.default_rng(seed)
    rows = cols = 3
    nodes = {(r, c): f"N{r}{c}" for r in range(rows) for c in range(cols)}

    edge_lbl = {}
    def lbl(a, b):
        key = tuple(sorted([a, b]))
        if key not in edge_lbl:
            edge_lbl[key] = f"e_{key[0][1:]}_{key[1][1:]}"
        return edge_lbl[key]

    tensors = {}
    for r in range(rows):
        for c in range(cols):
            n = nodes[(r, c)]
            neigh_labels = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    neigh_labels.append(lbl(n, nodes[(rr, cc)]))
            shape = tuple([dim] * len(neigh_labels))
            data = rng.normal(loc=1.0, scale=0.1, size=shape) + 1e-3
            tensors[n] = (data, neigh_labels)
            G.add_node(n)
    for (a, b) in edge_lbl:
        G.add_edge(a, b)
    return G, tensors


def exact_Z(graph, tensors) -> complex:
    return complex(contract_tensor_network(graph, tensors))


# ---- CSV writing -----------------------------------------------------------

CSV_FIELDS = [
    "method", "network_type", "dim", "size", "seed",
    "sign_flip_prob", "phase_strength",
    "basis_mode", "subset_size", "lambda_schedule",
    "K_lambda", "R", "C_mix", "M_phase", "C_phase",
    "A_beta", "B_chains", "C_updates",
    "n_G_evals", "time_seconds",
    "Z_true_real", "Z_true_imag",
    "Z_est_real", "Z_est_imag",
    "rel_error", "abs_error",
    "A_abs_est", "phase_est_real", "phase_est_imag", "abs_phase",
    "edge_abs_phase_if_available", "S_proj_over_m1_if_available",
    "mean_accept_rate", "min_ess", "mean_ess", "var_log_weights",
    "se_A", "se_S", "se_Z_delta", "n_zero_A_abs_chains",
]


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def base_row(method: str, network_type: str, dim: int, size: int, seed: int,
             sign_p: float, phase_s: float) -> Dict:
    return {
        "method": method,
        "network_type": network_type,
        "dim": dim,
        "size": size,
        "seed": seed,
        "sign_flip_prob": sign_p,
        "phase_strength": phase_s,
        "basis_mode": "",
        "subset_size": "",
        "lambda_schedule": "",
        "K_lambda": "",
        "R": "",
        "C_mix": "",
        "M_phase": "",
        "C_phase": "",
        "A_beta": "",
        "B_chains": "",
        "C_updates": "",
        "n_G_evals": "",
        "time_seconds": "",
        "edge_abs_phase_if_available": "",
        "S_proj_over_m1_if_available": "",
        "mean_accept_rate": "",
        "min_ess": "",
        "mean_ess": "",
        "var_log_weights": "",
    }


def fill_Z(row: Dict, Z_true: complex, Z_hat: complex) -> Dict:
    err = abs(Z_hat - Z_true)
    rel = err / abs(Z_true) if abs(Z_true) > 0 else float("inf")
    row["Z_true_real"] = Z_true.real
    row["Z_true_imag"] = Z_true.imag
    row["Z_est_real"] = Z_hat.real
    row["Z_est_imag"] = Z_hat.imag
    row["rel_error"] = rel
    row["abs_error"] = err
    return row
