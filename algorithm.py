"""
Tensor Network Contraction with Annealed Importance Sampling (AIS)
Authors: Sreevardhan Atyam, Anitej Chanda, Aniket Deshpande, Edgar Solomonik.
University of Illinois Urbana-Champaign
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class TensorNetwork:
    """
    Tensor Network object representing a closed network.
    Stores tensors, connectivity, index dimensions, and incident tensors.
    """
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
    """
    Evaluate the total weight ψ(c) = ∏_v M[v]_c for a given configuration.
    """
    result = 1.0
    for _, (tensor, inds) in network.tensors.items():
        key = tuple(config[idx] for idx in inds)
        result *= tensor[key]
    return result


def update_edge(network, config, idx, beta=1.0):
    """
    Perform a single-site Glauber update on index 'idx' at inverse temperature beta.
    """
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


def estimate_contraction(network, betas, iters=10000, burns=1000, verbose=False,
                         return_traj=False, return_weights=False):
    """
    Perform AIS with Glauber MCMC updates to estimate the TN contraction.
    """
    config = {idx: np.random.randint(dim) for idx, dim in network.index_dims.items()}
    logZ = 0.0
    logZ_traj = []
    all_weights = []

    for i in range(1, len(betas)):
        beta_prev, beta_curr = betas[i - 1], betas[i]
        delta_beta = beta_curr - beta_prev
        weights = []

        for t in range(iters):
            idx = np.random.choice(list(network.index_dims.keys()))
            update_edge(network, config, idx, beta=beta_curr)

            if t >= burns:
                w = evaluate_config(network, config)
                if w > 0:
                    weights.append(w ** (-delta_beta))

        weights = np.array(weights)
        ratio_est = np.mean(weights)
        logZ -= np.log(ratio_est)
        if return_traj:
            logZ_traj.append(logZ)
        if return_weights:
            all_weights.append(weights)

        if verbose:
            print(f"[β = {beta_curr:.2f}] E[ψ^(-Δβ)] = {ratio_est:.6f}")

    Z_est = np.exp(logZ) * np.prod(list(network.index_dims.values()))

    results = [Z_est]
    if return_traj:
        results.append(logZ_traj)
    if return_weights:
        results.append(all_weights)

    return tuple(results) if len(results) > 1 else results[0]


def run_multiple_chains(network, betas, n_chains=5, iters=10000, burns=1000, Z_true=None):
    """
    Run multiple AIS chains to compute statistics and visualize convergence.
    """
    estimates, logZ_trajectories, weight_variances = [], [], []

    for seed in range(n_chains):
        np.random.seed(seed)
        Z_est, logZ_traj, all_weights = estimate_contraction(
            network, betas, iters, burns, return_traj=True, return_weights=True
        )
        estimates.append(Z_est)
        logZ_trajectories.append(logZ_traj)
        rel_vars = [np.std(w) / np.mean(w) if np.mean(w) > 0 else 0 for w in all_weights]
        weight_variances.append(rel_vars)

    # plot convergence
    plt.figure(figsize=(12, 4))
    for i, traj in enumerate(logZ_trajectories):
        plt.plot(betas[1:], traj, label=f'chain {i+1}', alpha=0.8)
    plt.xlabel('beta')
    plt.ylabel('accumulated log Z')
    plt.title('log Z convergence')
    plt.grid(True)
    plt.legend()
    plt.show()

    # plot weight variance
    plt.figure(figsize=(12, 4))
    for i, rel_var in enumerate(weight_variances):
        plt.plot(betas[1:], rel_var, label=f'chain {i+1}', alpha=0.8)
    plt.xlabel('beta')
    plt.ylabel('relative std of weights')
    plt.title('variance of weights per beta')
    plt.grid(True)
    plt.legend()
    plt.show()

    # plot relative error if known
    if Z_true is not None:
        errors = [abs(est - Z_true) / abs(Z_true) for est in estimates]
        plt.figure(figsize=(7, 5))
        plt.plot(range(1, n_chains + 1), errors, 'o-')
        plt.xlabel('chain index')
        plt.ylabel('relative error')
        plt.title('relative error across chains')
        plt.grid(True)
        plt.show()

    return np.mean(estimates), np.std(estimates)


def test_trace_ABCD(dim=3, betas=np.linspace(0, 1, 100), n_chains=5, iters=20000, burns=1000):
    """
    Example test on a 2x2 ring: contraction of Tr(ABCD) with random positive matrices.
    """
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])

    A, B, C, D = [np.random.rand(dim, dim) + 0.1 for _ in range(4)]
    tensors = {
        'A': (A, ['i', 'j']),
        'B': (B, ['j', 'k']),
        'C': (C, ['k', 'l']),
        'D': (D, ['l', 'i'])
    }

    true_trace = np.trace(A @ B @ C @ D)
    tn = TensorNetwork(G, tensors)
    mean_Z, std_Z = run_multiple_chains(tn, betas, n_chains, iters, burns, Z_true=true_trace)

    print(f"\nestimated tr(ABCD): {mean_Z:.6f} ± {std_Z:.6f}")
    print(f"true tr(ABCD): {true_trace:.6f}")


if __name__ == "__main__":
    test_trace_ABCD()