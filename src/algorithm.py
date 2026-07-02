"""
Approximate Tensor Network Contraction with Annealed Importance Sampling (AIS)
Authors: Sreevardhan Atyam, Anitej Chanda, Aniket Deshpande, Qizhao Huang, Edgar Solomonik.
University of Illinois Urbana-Champaign
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from datetime import datetime
from typing import Optional, Tuple

sns.set_theme(style="whitegrid")


class TensorNetwork:
    def __init__(self, graph, tensors):
        self.graph = graph
        self.tensors = tensors
        self.index_dims = {}
        self.index_to_tensors = {}
        for name, (tensor, indices) in tensors.items():
            for idx, dim in zip(indices, tensor.shape):
                if idx in self.index_dims:
                    assert self.index_dims[idx] == dim
                else:
                    self.index_dims[idx] = dim
                self.index_to_tensors.setdefault(idx, []).append((name, tensor, indices))
        self.col_of = {idx: c for c, idx in enumerate(self.index_dims)}


def evaluate_logpsi(network: TensorNetwork, configs: np.ndarray) -> np.ndarray:
    """Return log ψ(c) for each chain c, without clipping (numerically stable)."""
    log_out = np.zeros(len(configs), dtype=np.float64)
    for _, (tensor, inds) in network.tensors.items():
        keys = tuple(configs[:, network.col_of[i]] for i in inds)
        vals = tensor[keys].astype(np.float64, copy=False)  # shape (n_chains,)
        # if any vals can be 0, treat as -inf (zero weight)
        log_out += np.where(vals > 0.0, np.log(vals), -np.inf)
    return log_out


def evaluate_config(network: TensorNetwork, configs: np.ndarray) -> np.ndarray:
    """(Optional wrapper) ψ(c) in value-space. May underflow for large nets."""
    return np.exp(evaluate_logpsi(network, configs))


def update_edge(network: TensorNetwork, configs: np.ndarray, idx: str, beta: float = 1.0):
    """Gibbs update for index `idx` using log-space softmax for numerical stability."""
    dim = network.index_dims[idx]
    col = network.col_of[idx]
    n_chains = configs.shape[0]

    # accumulate log-probs: log p(a) ∝ beta * sum_f log f(a, neighbors)
    logp = np.zeros((n_chains, dim), dtype=np.float64)

    for _, tensor, inds in network.index_to_tensors[idx]:
        slc = [slice(None) if i == idx else configs[:, network.col_of[i]] for i in inds]
        vals = tensor[tuple(slc)]
        if vals.shape != (n_chains, dim):
            vals = vals.T
        vals = vals.astype(np.float64, copy=False)

        logp += beta * np.where(vals > 0.0, np.log(vals), -np.inf)

    # stable softmax row-wise
    row_max = np.max(logp, axis=1, keepdims=True)
    probs = np.empty_like(logp)

    finite = np.isfinite(row_max).reshape(-1)
    if np.any(finite):
        lp = logp[finite] - row_max[finite]
        p = np.exp(lp)
        p_sum = p.sum(axis=1, keepdims=True)
        probs[finite] = p / p_sum

    # if a row is all -inf (shouldn't happen with eps floors), fall back to uniform
    if np.any(~finite):
        probs[~finite] = 1.0 / dim

    # resample
    configs[:, col] = [np.random.choice(dim, p=probs[i]) for i in range(n_chains)]


def estimate_contraction(
    net: TensorNetwork,
    betas: np.ndarray,
    iters: int = 20000,
    burns: int = 1900,
    n_chains: int = 10,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard AIS: exactly one weight per chain per beta-step.
    Returns:
      Z_ests       – N_chains estimates of Z
      logZ_trajs   – list of length N_chains of log-Z trajectories
      weights_by_beta – array shape (K, N_chains) of incremental weights
    """
    K = len(betas) - 1
    index_list = list(net.index_dims)
    # initial random configurations
    configs = np.empty((n_chains, len(index_list)), dtype=int)
    for j, idx in enumerate(index_list):
        configs[:, j] = np.random.randint(0, net.index_dims[idx], size=n_chains)
    np.random.shuffle(configs.T)

    logZ_sums      = np.zeros(n_chains)
    logZ_trajs     = [ [] for _ in range(n_chains) ]
    weights_by_beta= np.zeros((K, n_chains))

    for k in range(1, len(betas)):
        b_prev, b_curr = betas[k-1], betas[k]
        db = b_curr - b_prev

        if verbose and (k % 10 == 0 or k == K):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] β-step {k}/{K} (β={b_curr:.4f})")

        # 1) mix **all** chains under π_{β_prev}
        for t in range(iters):
            idx = np.random.choice(index_list)
            update_edge(net, configs, idx, beta=b_prev)

        # 2) compute weights in log space (stable, no clipping bias)
        logpsi = evaluate_logpsi(net, configs)      # shape (n_chains,)
        log_w  = db * logpsi                        # incremental log-weights (stable)

        # compute w stably for storing/printing (avoids overflow in intermediate)
        m = np.max(log_w)
        w = np.exp(log_w - m) * np.exp(m)           # equals exp(log_w)

        # record
        weights_by_beta[k-1, :] = w
        logZ_sums += log_w
        for j in range(n_chains):
            logZ_trajs[j].append(logZ_sums[j])

        if verbose:
            print(f"    ⟨w⟩ = {w.mean():.3e},   std(log w) = {np.std(log_w):.3e}")

    # finish
    logZ0 = np.sum(np.log(list(net.index_dims.values())))
    Z_ests = np.exp(logZ_sums + logZ0)
    return Z_ests, np.array(logZ_trajs), weights_by_beta


def run_multiple_chains(
    tn: TensorNetwork,
    betas: np.ndarray,
    iters: int=20000,
    burns: int=1900,
    n_chains: int=10,
    verbose: bool=True,
    Z_true: float=None
) -> Tuple[float, float]:
    Z_ests, logZ_trajs, w_by_beta = estimate_contraction(
        tn, betas, iters=iters, burns=burns, n_chains=n_chains, verbose=verbose
    )

    # simple diagnostics
    # … you can reuse your run_diagnostics, adapting it to w_by_beta[k] being shape (n_chains,)
    return Z_ests.mean(), Z_ests.std(ddof=1)


def contract_tensor_network(graph, tensors):
    """
    Efficient full contraction via np.einsum.
    `tensors` is a dict: node_name -> (ndarray, [index labels]).
    """
    einsum_terms = []
    einsum_tensors = []
    index_map = {}
    chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    for _, (tensor, indices) in tensors.items():
        subs = []
        for idx in indices:
            if idx not in index_map:
                index_map[idx] = chars.pop(0)
            subs.append(index_map[idx])
        einsum_terms.append(''.join(subs))
        einsum_tensors.append(tensor)

    expr = ','.join(einsum_terms) + '->'
    return np.einsum(expr, *einsum_tensors, optimize='greedy')