"""
Approximate Tensor Network Contraction with Annealed Importance Sampling (AIS)
Authors: Sreevardhan Atyam, Anitej Chanda, Aniket Deshpande, Edgar Solomonik.
University of Illinois Urbana-Champaign
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from datetime import datetime
from typing import Optional, Tuple

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
        
        self.col_of = {idx: c for c, idx in enumerate(self.index_dims)}

def evaluate_config(network, configs):
    result = np.ones(len(configs))
    for name, (tensor, inds) in network.tensors.items():
        keys = tuple(configs[:, network.col_of[i]] for i in inds)
        result *= tensor[keys]
    return result

def update_edge(network, configs, idx, beta=1.0):
    dim = network.index_dims[idx]
    col = network.col_of[idx]
    n_chains = configs.shape[0]
    probs = np.ones((n_chains, dim))

    for _, tensor, inds in network.index_to_tensors[idx]:
        slc = [slice(None) if i == idx else configs[:, network.col_of[i]] for i in inds]
        arr_vals = tensor[tuple(slc)]
        if arr_vals.shape != (n_chains, dim):
            arr_vals = arr_vals.T
        probs *= arr_vals ** beta

    probs /= probs.sum(axis=1, keepdims=True)
    configs[:, col] = [np.random.choice(dim, p=probs[i]) for i in range(n_chains)]

def estimate_contraction(net, betas, iters=20000, burns=1900, n_chains=10, verbose=True):
    n_betas = len(betas)
    index_list = list(net.index_dims)
    n_sites = len(index_list)

    # init configs in a more structured way
    configs = np.tile(np.arange(3), (n_chains, int(np.ceil(n_sites / 3))))[:, :n_sites]
    np.random.shuffle(configs.T)
    
    logZ_sums = np.zeros(n_chains)
    logZ_trajs = [[] for _ in range(n_chains)]
    weights_by_beta = [[] for _ in range(n_betas - 1)]

    for step in range(1, n_betas):
        beta_prev, beta_curr = betas[step - 1], betas[step]
        delta_beta = beta_curr - beta_prev

        if verbose and (step % 10 == 0 or step == n_betas - 1):
            print(f'[{datetime.now().strftime("%H:%M:%S")}] beta step {step}/{n_betas - 1} ({beta_curr:.4f})')

        chain_weights = [[] for _ in range(n_chains)]

        for t in range(iters):
            idx = np.random.choice(index_list)
            update_edge(net, configs, idx, beta=beta_prev)

            if t >= burns:
                psi_vals = evaluate_config(net, configs)
                weights = np.where(psi_vals > 1e-30, psi_vals ** (delta_beta), 0.0)
                for j in range(n_chains):
                    chain_weights[j].append(weights[j])

        for j in range(n_chains):
            w = chain_weights[j]
            mw = np.mean(w) if w else 0.0
            log_rho = np.logaddexp.reduce(np.log(w)) - np.log(len(w))
            logZ_sums[j] += log_rho
            logZ_trajs[j].append(logZ_sums[j])
            weights_by_beta[step - 1].append(np.asarray(w))

        if verbose:
            mean_log_rho = np.mean([np.log(np.mean(w)) if np.mean(w) > 0 else -np.inf for w in chain_weights])
            mean_w = np.mean([np.mean(w) for w in chain_weights])
            print(f"[beta={beta_curr:.3f}]  mean log rho = {mean_log_rho:+.3e}  | <w> mean = {mean_w:.4e}")

    log_size = np.sum(np.log(list(net.index_dims.values())))
    Z_ests = np.exp(logZ_sums + log_size)
    return Z_ests, logZ_trajs, weights_by_beta

def run_diagnostics(betas, logZ_trajectories, weights_by_beta, Z_ests, Z_true=None):
    n_chains = len(Z_ests)

    # logZ convergence
    plt.figure(figsize=(12, 3.8))
    for k, traj in enumerate(logZ_trajectories, 1):
        plt.plot(betas[1:], traj, label=f"chain {k}", alpha=0.8)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\log Z$")
    plt.title("log-Z convergence across chains")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # relative std dev of weights
    plt.figure(figsize=(12, 3.8))
    for k in range(n_chains):
        rel_sig = []
        for weights in weights_by_beta:
            w = weights[k]
            rel = np.std(w) / np.mean(w) if np.mean(w) > 0 else 0
            rel_sig.append(rel)
        plt.plot(betas[1:], rel_sig, label=f"chain {k+1}", alpha=0.8)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"rel $\sigma(w)$")
    plt.title("weight dispersion vs beta")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # log-weight variance
    log_var_per_beta = []
    for beta_weights in weights_by_beta:
        log_vars = [np.var(np.log(np.clip(w, 1e-12, None))) for w in beta_weights]
        log_var_per_beta.append(np.mean(log_vars))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.lineplot(x=betas[1:], y=log_var_per_beta, marker='o', ax=ax)
    ax.axhline(1.0, ls='--', color='gray', label='Variance Threshold')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'Variance of $\log w$')
    ax.set_title("Variance of Log-Weights vs Beta Step")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # error bars
    est_mean, est_std = np.mean(Z_ests), np.std(Z_ests)
    print(rf"\nFinal relative variance of $\hat Z$: {est_std / est_mean:.4e}")

    if Z_true is not None:
        rel_errs = [abs(z - Z_true) / abs(Z_true) for z in Z_ests]
        plt.figure(figsize=(5.5, 4))
        plt.bar(range(1, n_chains + 1), rel_errs)
        plt.xlabel("chain")
        plt.ylabel("rel error")
        plt.title("final relative error per chain")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"True  Z   : {Z_true:.6f}")
        print(f"Mean  Ẑ  : {est_mean:.6f}  ± {est_std:.6f}")
        print(f"Mean rel error: {np.mean(rel_errs):.2%}")

def run_multiple_chains(tn: TensorNetwork,
                       betas: np.ndarray,
                       n_chains: Optional[int] = None,
                       iters: Optional[int] = None,
                       burns: Optional[int] = None,
                       Z_true: Optional[float] = None,
                       show_diagnostics: Optional[bool] = None) -> Tuple[float, float]:
    # use config values if not specified
    n_chains = n_chains or tn.config.n_chains
    iters = iters or tn.config.iters
    burns = burns or tn.config.burns
    show_diagnostics = show_diagnostics if show_diagnostics is not None else tn.config.show_diagnostics

    # get dimension from the first tensor in the network
    dim = next(iter(tn.tensors.values()))[0].shape[0]

    # init configurations
    n_sites = len(tn.graph.nodes())
    configs = np.tile(np.arange(dim), (n_chains, int(np.ceil(n_sites / dim))))[:, :n_sites]
    
    if show_diagnostics:
        print(f"Running {n_chains} AIS chains vectorized")
    Z_ests, logZ_trajectories, weights_by_beta = estimate_contraction(
        tn, betas, iters, burns, n_chains=n_chains, verbose=show_diagnostics
    )

    if show_diagnostics:
        run_diagnostics(betas, logZ_trajectories, weights_by_beta, Z_ests, Z_true=Z_true)
    return np.mean(Z_ests), np.std(Z_ests)

def contract_tensor_network(graph, tensors):
    """
    Efficient full contraction using einsum.
    tensors: dict[node_name] = (ndarray, [str indices])
    """
    einsum_terms = []
    einsum_tensors = []
    for _, (tensor, indices) in tensors.items():
        einsum_terms.append(''.join(indices))
        einsum_tensors.append(tensor)
    einsum_expr = ','.join(einsum_terms)
    return np.einsum(einsum_expr, *einsum_tensors)