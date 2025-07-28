import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from src.algorithm import TensorNetwork, run_multiple_chains

# contract full tensor network using np.einsum
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

# build 3x3 grid tensor network
def build_3x3_grid_test(dim=3):
    G = nx.Graph()
    tensors = {}
    grid_size = 3
    node_names = {(i, j): f"T{i}{j}" for i in range(grid_size) for j in range(grid_size)}

    def edge_index(i1, j1, i2, j2):
        return f"{i1}{j1}_{i2}{j2}"

    for i in range(grid_size):
        for j in range(grid_size):
            name = node_names[(i, j)]
            neighbors = []

            if j + 1 < grid_size:
                nbr = node_names[(i, j+1)]
                idx = edge_index(i, j, i, j+1)
                G.add_edge(name, nbr)
                neighbors.append(idx)

            if i + 1 < grid_size:
                nbr = node_names[(i+1, j)]
                idx = edge_index(i, j, i+1, j)
                G.add_edge(name, nbr)
                neighbors.append(idx)

            if j > 0:
                neighbors.append(edge_index(i, j-1, i, j))
            if i > 0:
                neighbors.append(edge_index(i-1, j, i, j))

            neighbors = sorted(neighbors)
            shape = (dim,) * len(neighbors)
            data = np.random.rand(*shape) + 1e-6  # positive entries ∈ (1e-6, 1)
            tensors[name] = (data, neighbors)

    return G, tensors

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
        print(f"Mean rel error: {np.mean(rel_errs):.8%}")

# test on 3x3 grid tensor network
def test_trace_3x3_grid(dim=3,
                        betas=np.linspace(0, 1, 200),
                        n_chains=5,
                        iters=20000,
                        burns=1900,
                        show_diagnostics=True):
    if show_diagnostics:
        print("\n>>> Building 3x3 grid tensor network")

    # create 3x3 grid tensor network
    G = nx.Graph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'),
        ('D', 'E'), ('E', 'F'),
        ('G', 'H'), ('H', 'I'),
        ('A', 'D'), ('D', 'G'),
        ('B', 'E'), ('E', 'H'),
        ('C', 'F'), ('F', 'I'),
    ])

    # Initialize tensors with normal distribution for better numerical stability
    tensors = {}
    index_order = {
        'A': ['i', 'j'], 'B': ['j', 'k'], 'C': ['k', 'l'],
        'D': ['i', 'm'], 'E': ['j', 'n'], 'F': ['k', 'o'],
        'G': ['m', 'p'], 'H': ['n', 'p'], 'I': ['o', 'p']
    }
    for node, inds in index_order.items():
        data = np.random.normal(loc=1.0, scale=0.01, size=(dim, dim)) + 1e-6
        tensors[node] = (data, inds)

    # exact contraction: trace of product
    A, B, C = [tensors[k][0] for k in ['A', 'B', 'C']]
    D, E, F = [tensors[k][0] for k in ['D', 'E', 'F']]
    G, H, I = [tensors[k][0] for k in ['G', 'H', 'I']]
    TRUE_Z = np.einsum('ij,jk,kl,im,jn,ko,mp,np,op->', A, B, C, D, E, F, G, H, I)
    if show_diagnostics:
        print(f"True Z (3x3 grid): {TRUE_Z:.6f}")

    tn = TensorNetwork(G, tensors)

    if show_diagnostics:
        print("\n>>> Running MCMC chains")
    mean_Z, std_Z = run_multiple_chains(
        tn, betas,
        n_chains=n_chains,
        iters=iters,
        burns=burns,
        Z_true=TRUE_Z if show_diagnostics else None,
        show_diagnostics=show_diagnostics
    )

    rel_error = abs(mean_Z - TRUE_Z) / abs(TRUE_Z)
    if show_diagnostics:
        print("\n================== Final Summary ==================")
        print(f"True       Z : {TRUE_Z:.6f}")
        print(f"Estimated  Z : {mean_Z:.6f} ± {std_Z:.6f}")
        print(f"Relative Error: {rel_error:.8%}")
        print("====================================================")

    return mean_Z, std_Z, rel_error

if __name__ == "__main__":
    test_trace_3x3_grid()