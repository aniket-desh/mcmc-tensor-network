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


def evaluate_config(network: TensorNetwork, configs: np.ndarray) -> np.ndarray:
    out = np.ones(len(configs))
    for name, (tensor, inds) in network.tensors.items():
        keys = tuple(configs[:, network.col_of[i]] for i in inds)
        out *= tensor[keys]
    return np.clip(out, 1e-30, None)


def update_edge(network: TensorNetwork, configs: np.ndarray, idx: str, beta: float = 1.0):
    dim = network.index_dims[idx]
    col = network.col_of[idx]
    n_chains = configs.shape[0]
    probs = np.ones((n_chains, dim))
    # multiply in each tensor factor touching idx
    for _, tensor, inds in network.index_to_tensors[idx]:
        slc = [slice(None) if i == idx else configs[:, network.col_of[i]] for i in inds]
        vals = tensor[tuple(slc)]
        if vals.shape != (n_chains, dim):
            vals = vals.T
        probs *= np.clip(vals, 1e-30, None) ** beta
    probs /= probs.sum(axis=1, keepdims=True)
    # resample that index in all chains
    configs[:, col] = [
        np.random.choice(dim, p=probs[i]) for i in range(n_chains)
    ]


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

        # 2) one weight per chain
        psi_vals = evaluate_config(net, configs)                 # shape (n_chains,)
        w      = np.clip(psi_vals, 1e-30, None) ** db            # incremental weights
        log_w  = np.log(w)

        # record
        weights_by_beta[k-1,:] = w
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