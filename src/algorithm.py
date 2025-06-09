"""
Tensor Network Contraction with Annealed Importance Sampling (AIS)
Authors: Sreevardhan Atyam, Anitej Chanda, Aniket Deshpande, Edgar Solomonik.
University of Illinois Urbana-Champaign
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

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
    for _, (tensor, inds) in network.tensors.items():
        keys = tuple(configs[i, network.col_of[i]] for i in inds)
        result *= tensor[keys]
    return result

def update_edge(network, configs, idx, beta=1.0):
    dim = network.index_dims[idx]
    col = network.col_of[idx]
    n_chains = configs.shape[0]

    probs = np.ones((n_chains, dim))
    for _, tensor, inds in network.index_to_tensors[idx]:
        slc = []
        for i in inds:
            if i == idx:
                slc.append(slice(None))
            else:
                slc.append(configs[:, network.col_of[i]])
        arr_vals = tensor[tuple(slc)]
        if arr_vals.shape != (n_chains, dim):
            arr_vals = arr_vals.reshape(n_chains, dim)
        probs *= arr_vals ** beta

    probs /= probs.sum(axis=1, keepdims=True)
    new_vals = [np.random.choice(dim, p=probs[i]) for i in range(n_chains)]
    configs[:, col] = new_vals


def estimate_contraction(net, betas, iters=10_000, burns=1_000, n_rounds=5, verbose=True):
    logZ_trajs = [[] for _ in range(n_rounds)]
    weights_by_beta = [[] for _ in range(len(betas)-1)]
    logZ_sums = np.zeros(n_rounds)

    index_list = list(net.index_dims)
    total_steps = len(betas) - 1
    dim_size = len(index_list)

    configs = np.random.randint(0, 3, size=(n_rounds, dim_size))

    for i in range(1, len(betas)):
        beta_prev, beta_curr = betas[i-1], betas[i]
        delta_beta = beta_curr - beta_prev

        if verbose and (i % 10 == 0 or i == total_steps):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f'[{timestamp}] beta step {i}/{total_steps} ({beta_curr:.4f}) ')

        all_weights = [[] for _ in range(n_rounds)]

        for t in range(iters):
            idx = np.random.choice(index_list)
            update_edge(net, configs, idx, beta=beta_prev)
            if t >= burns:
                psi_vals = evaluate_config(net, configs)
                valid = psi_vals > 1e-30
                weights = np.where(valid, psi_vals ** delta_beta, 0.0)
                for j in range(n_rounds):
                    all_weights[j].append(weights[j])

        mean_weights = [np.mean(w) if len(w) > 0 else 0.0 for w in all_weights]
        log_rhos     = [np.log(mw) if mw > 0 else -np.inf for mw in mean_weights]

        for j in range(n_rounds):
            logZ_sums[j] += log_rhos[j]
            logZ_trajs[j].append(logZ_sums[j])
            weights_by_beta[i-1].append(np.asarray(all_weights[j]))

        if verbose:
            print(f"[beta={beta_curr:.3f}]  mean log rho = {np.mean(log_rhos):+.3e}  | <w> mean = {np.mean(mean_weights):.4e}")

    log_size = np.sum(np.log(list(net.index_dims.values())))
    Z_ests = np.exp(logZ_sums + log_size)
    return Z_ests, logZ_trajs, weights_by_beta


# run several independent AIS chains and visualize convergence
def run_multiple_chains(net        : TensorNetwork,
                        betas      : np.ndarray,
                        n_chains   : int  = 5,
                        iters      : int  = 10_000,
                        burns      : int  = 1_000,
                        n_rounds   : int  = 5,
                        Z_true     : float|None = None,
                        n_workers  : int  = None):

    print(f"Running {n_chains} AIS chains vectorized")
    Z_ests, logZ_trajectories, weights_by_beta = estimate_contraction(
        net, betas, iters, burns, n_rounds=n_chains, verbose=True
    )

    # ---------- 1) cumulative log-Z trajectories ----------------------- #
    plt.figure(figsize=(12, 3.8))
    for k, traj in enumerate(logZ_trajectories, 1):
        plt.plot(betas[1:], traj, label=f"chain {k}", alpha=0.8)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\log Z$ (cumulative.)")
    plt.title("log-Z convergence across chains")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # ---------- 2) relative sigma(w) per beta -------------------------------- #
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

    # ---------- 3) final-estimate statistics -------------------------- #
    est_mean, est_std = np.mean(Z_ests), np.std(Z_ests)
    rel_var = est_std / est_mean if est_mean > 0 else float("nan")
    print(rf"\nFinal relative variance of $\hat Z$: {rel_var:.4e}")

    if Z_true is not None:
        rel_errs = [abs(z - Z_true)/abs(Z_true) for z in Z_ests]
        plt.figure(figsize=(5.5, 4))
        plt.bar(range(1, n_chains+1), rel_errs)
        plt.xlabel("chain"); plt.ylabel("rel error")
        plt.title("final relative error per chain")
        plt.grid(True); plt.tight_layout(); plt.show()

        print(f"True  Z   : {Z_true:.6f}")
        print(f"Mean  Ẑ  : {est_mean:.6f}  ± {est_std:.6f}")
        print(f"Mean rel error: {np.mean(rel_errs):.2%}")

    return est_mean, est_std