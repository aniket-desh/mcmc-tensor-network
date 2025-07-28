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
                    assert self.index_dims[idx] == dim, f"Index {idx} dimension mismatch."
                else:
                    self.index_dims[idx] = dim
                self.index_to_tensors.setdefault(idx, []).append((name, tensor, indices))
        
        # map each index label to a column in the config array
        self.col_of = {idx: c for c, idx in enumerate(self.index_dims)}


def evaluate_config(network: TensorNetwork, configs: np.ndarray) -> np.ndarray:
    """
    Given an array of configurations (shape [n_chains, n_sites]),
    multiply together the tensor entries at those index values.
    Returns a vector of length n_chains.
    """
    out = np.ones(configs.shape[0])
    for name, (tensor, inds) in network.tensors.items():
        keys = tuple(configs[:, network.col_of[i]] for i in inds)
        out *= tensor[keys]
    return np.clip(out, 1e-30, None)


def update_edge(network: TensorNetwork, configs: np.ndarray, idx: str, beta: float = 1.0):
    """
    Single-site Glauber update on index 'idx' for all chains in 'configs',
    targeting the distribution \pi_\beta(x) \propto \psi(x)^\beta.
    """
    dim = network.index_dims[idx]
    col = network.col_of[idx]
    n_chains = configs.shape[0]

    # build the unnormalized probability table for each chain, each possible label
    probs = np.ones((n_chains, dim))
    for _, tensor, inds in network.index_to_tensors[idx]:
        # slice out the tensor values along this index for every chain
        slc = [slice(None) if i == idx else configs[:, network.col_of[i]] for i in inds]
        vals = tensor[tuple(slc)]
        if vals.shape != (n_chains, dim):
            vals = vals.T
        probs *= np.clip(vals, 1e-30, None) ** beta

    # normalize and resample
    probs /= probs.sum(axis=1, keepdims=True)
    for j in range(n_chains):
        configs[j, col] = np.random.choice(dim, p=probs[j])


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
      Z_ests         – array of length n_chains: individual estimates of Z
      logZ_trajs     – list of length n_chains of cumulative log Z increments at each step
      weights_by_beta– ndarray of shape (K, n_chains) of incremental weights w_k,j
    """
    K = len(betas) - 1
    index_list = list(net.index_dims)
    n_sites = len(index_list)

    # initialize each chain to a random configuration
    configs = np.empty((n_chains, n_sites), dtype=int)
    for j, idx in enumerate(index_list):
        configs[:, j] = np.random.randint(0, net.index_dims[idx], size=n_chains)
    np.random.shuffle(configs.T)

    logZ_sums       = np.zeros(n_chains)
    logZ_trajs      = [ [] for _ in range(n_chains) ]
    weights_by_beta = np.zeros((K, n_chains))

    for k in range(1, len(betas)):
        b_prev, b_curr = betas[k-1], betas[k]
        delta_beta = b_curr - b_prev

        if verbose and (k % 10 == 0 or k == K):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] β-step {k}/{K} (β={b_curr:.4f})")

        # 1) mix each chain under π_{β_prev} via iters single-site updates
        for t in range(iters):
            idx = np.random.choice(index_list)
            update_edge(net, configs, idx, beta=b_prev)

        # 2) one importance weight per chain
        psi_vals = evaluate_config(net, configs)  # length n_chains
        w        = psi_vals ** delta_beta         # incremental weights
        log_w    = np.log(w)

        # record into arrays
        weights_by_beta[k-1, :] = w
        logZ_sums += log_w
        for j in range(n_chains):
            logZ_trajs[j].append(logZ_sums[j])

        if verbose:
            print(f"    ⟨w⟩ = {w.mean():.3e},   std(log w) = {np.std(log_w):.3e}")

    # final normalization by the partition function at β=0
    logZ0 = np.sum(np.log(list(net.index_dims.values())))
    Z_ests = np.exp(logZ_sums + logZ0)
    return Z_ests, np.array(logZ_trajs), weights_by_beta


def run_multiple_chains(
    tn: TensorNetwork,
    betas: np.ndarray,
    iters: int = 20000,
    burns: int = 1900,
    n_chains: int = 10,
    verbose: bool = True,
    Z_true: Optional[float] = None
) -> Tuple[float, float]:
    """
    Wrapper to run multiple AIS chains and optionally show diagnostics.
    Returns (mean Z_hat, std Z_hat).
    """
    Z_ests, logZ_trajs, w_by_beta = estimate_contraction(
        tn, betas, iters=iters, burns=burns, n_chains=n_chains, verbose=verbose
    )

    if verbose:
        # you can reuse or simplify run_diagnostics here
        print(f"\nEstimated Z mean ± std: {Z_ests.mean():.6g} ± {Z_ests.std(ddof=1):.6g}")
        if Z_true is not None:
            rel_errs = np.abs(Z_ests - Z_true) / np.abs(Z_true)
            print(f"Mean relative error: {rel_errs.mean():.6%}")

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