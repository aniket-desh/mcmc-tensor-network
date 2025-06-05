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

def evaluate_config(network, config):
    result = 1.0
    for _, (tensor, inds) in network.tensors.items():
        key = tuple(config[idx] for idx in inds)
        result *= tensor[key]
    return result

def update_edge(network, config, idx, beta=1.0):
    tensors = network.index_to_tensors[idx]
    dim = network.index_dims[idx]
    probs = np.ones(dim)

    for _, arr, inds in tensors:
        slc = [config[i] if i != idx else slice(None) for i in inds]
        probs *= arr[tuple(slc)] ** beta

    probs_sum = probs.sum()
    if probs_sum > 0:
        probs /= probs_sum
    else:
        probs = np.ones(dim) / dim

    config[idx] = np.random.choice(dim, p=probs)

def estimate_contraction(net, betas, iters=10000, burns=1000, n_rounds=5, verbose=False):
    logZ_sum = 0.0
    logZ_traj = []
    weights_by_beta = []

    index_list = list(net.index_dims)
    total_steps = len(betas) - 1

    for i in range(1, len(betas)):
        beta_prev, beta_curr = betas[i-1], betas[i]
        delta_beta = beta_curr - beta_prev
        all_weights = []

        if verbose and (i % 10 == 0 or i == total_steps):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f'[{timestamp}] beta step {i}/{total_steps} ({beta_curr:.4f})')

        for _ in range(n_rounds):
            cfg = {idx: np.random.randint(d) for idx, d in net.index_dims.items()}

            for t in range(iters):
                idx = np.random.choice(index_list)
                update_edge(net, cfg, idx, beta=beta_prev)
                if t >= burns:
                    psi = evaluate_config(net, cfg)
                    weight = psi ** delta_beta
                    all_weights.append(weight)

        if len(all_weights) == 0:
            log_rho = np.NINF
            all_weights = [0.0]
            if verbose:
                print(f'[beta={beta_curr:.3f}]  NO VALID WEIGHTS SAMPLED')
        else:
            log_rho = np.log(np.mean(all_weights))

        logZ_sum += log_rho
        logZ_traj.append(logZ_sum)
        weights_by_beta.append(np.asarray(all_weights))

        if verbose:
            print(f"[beta={beta_curr:.3f}]  log rho={log_rho:+.3e} | "
                  f"cum log Z={logZ_sum:+.3e} | "
                  f"<w>={np.mean(all_weights):.4e}  var(w)={np.var(all_weights):.4e}")

    log_size = np.sum(np.log(list(net.index_dims.values())))
    Z_est = np.exp(logZ_sum + log_size)
    return Z_est, logZ_traj, weights_by_beta

def run_multiple_chains(net, betas, n_chains=5, iters=10000, burns=1000, n_rounds=5, Z_true=None):
    estimates = []
    logZ_trajectories = []
    weight_variances = []

    for c in range(n_chains):
        np.random.seed(c)
        print(f"\n==== chain {c+1}/{n_chains} ====")

        Z_hat, log_traj, w_by_beta = estimate_contraction(
            net, betas, iters, burns, n_rounds, verbose=False
        )

        estimates.append(Z_hat)
        logZ_trajectories.append(log_traj)

        rel_sig = [ (np.std(w)/np.mean(w)) if np.mean(w) > 0 else 0.0 for w in w_by_beta ]
        weight_variances.append(rel_sig)

    plt.figure(figsize=(12, 3.8))
    for k, traj in enumerate(logZ_trajectories, 1):
        plt.plot(betas[1:], traj, label=f"chain {k}", alpha=0.8)
    plt.xlabel(r"$\\beta$")
    plt.ylabel(r"$\\log Z$ (cum.)")
    plt.title("log-Z convergence across chains")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(12, 3.8))
    for k, rel_sig in enumerate(weight_variances, 1):
        plt.plot(betas[1:], rel_sig, label=f"chain {k}", alpha=0.8)
    plt.xlabel(r"$\\beta$")
    plt.ylabel(r"rel $\\sigma(w)$")
    plt.title("weight dispersion vs beta")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    est_mean, est_std = np.mean(estimates), np.std(estimates)
    rel_var = est_std / est_mean if est_mean > 0 else float("nan")
    print(rf"\nFinal relative variance of $\\hat Z$: {rel_var:.4e}")

    if Z_true is not None:
        rel_errs = [abs(z - Z_true)/abs(Z_true) for z in estimates]
        plt.figure(figsize=(5.5, 4))
        plt.bar(range(1, n_chains+1), rel_errs)
        plt.xlabel("chain"); plt.ylabel("rel error")
        plt.title("final relative error per chain")
        plt.grid(True); plt.tight_layout(); plt.show()

        print(f"True  Z   : {Z_true:.6f}")
        print(f"Mean  Ẑ  : {est_mean:.6f}  ± {est_std:.6f}")
        print(f"Mean rel error: {np.mean(rel_errs):.2%}")

    return est_mean, est_std