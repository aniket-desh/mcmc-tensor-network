# Test AIS on 3x3 grid tensor network with exponentially-decaying diagonal
import numpy as np
import networkx as nx
from src.algorithm import TensorNetwork, run_multiple_chains, estimate_contraction


def contract_tensor_network(graph, tensors):
    """
    Efficient full contraction using einsum.
    tensors: dict[node_name] = (ndarray, [str indices])
    """
    einsum_terms = []
    einsum_tensors = []
    index_map = {}
    chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    for _, (tensor, indices) in tensors.items():
        subs = []
        for idx in indices:
            if idx not in index_map:
                if not chars:
                    raise ValueError("[error] ran out of characters for einsum indices")
                index_map[idx] = chars.pop(0)
            subs.append(index_map[idx])
        einsum_terms.append(''.join(subs))
        einsum_tensors.append(tensor)

    expr = ','.join(einsum_terms) + '->'
    return np.einsum(expr, *einsum_tensors, optimize='greedy')


def build_3x3_grid_exp_diag(dim=3, alpha=0.6, eps=1e-6, normalize=False, seed=42):
    """
    3x3 grid with exponentially decaying diagonal entries.
    Each tensor T has shape (dim,)^(degree(node)).
      T[k, k, ..., k] = exp(-alpha * k)  for k=0..dim-1
      T[off-diagonal] = eps

    Args:
        dim: Dimension of each index
        alpha: Decay rate in exp(-alpha * k)
        eps: Small positive floor for all off-diagonal entries
        normalize: If True, scale diag so its sum is 1 (per tensor)
        seed: RNG seed for deterministic builds

    Returns:
        G: NetworkX graph
        tensors: dict[str, tuple[np.ndarray, list[str]]]
    """
    np.random.seed(seed)

    G = nx.Graph()
    tensors = {}
    grid_size = 3
    node_names = {(i, j): f"T{i}{j}" for i in range(grid_size) for j in range(grid_size)}

    def edge_index(i1, j1, i2, j2):
        return "_".join(sorted([f"{i1}{j1}", f"{i2}{j2}"]))

    # Precompute diagonal values
    diag_vals = np.exp(-alpha * np.arange(dim))
    if normalize:
        s = diag_vals.sum()
        if s > 0:
            diag_vals = diag_vals / s

    for i in range(grid_size):
        for j in range(grid_size):
            name = node_names[(i, j)]

            nbrs = []
            if i > 0:                nbrs.append((i - 1, j))
            if i < grid_size - 1:    nbrs.append((i + 1, j))
            if j > 0:                nbrs.append((i, j - 1))
            if j < grid_size - 1:    nbrs.append((i, j + 1))

            indices = sorted(edge_index(i, j, ni, nj) for (ni, nj) in nbrs)
            G.add_node(name)
            for ni, nj in nbrs:
                G.add_edge(name, node_names[(ni, nj)])

            shape = (dim,) * len(indices)
            rank = len(shape)

            # Hyper-diagonal tensor with exponential decay + epsilon elsewhere
            T = np.full(shape, eps, dtype=float)
            for k in range(dim):
                idx = (k,) * rank
                T[idx] = diag_vals[k]

            tensors[name] = (T, indices)

    return G, tensors


def make_logspace_betas(A):
    """Generate logspace beta schedule for better early-beta resolution."""
    betas = 1.0 - np.logspace(0, np.log10(1e-6), A)
    betas[0] = 0.0
    betas[-1] = 1.0
    return np.sort(betas)


def test_trace_3x3_grid_exp_diag(dim=3,
                                  A=200,
                                  B=400,
                                  C=200,
                                  alpha=0.6,
                                  show_diagnostics=True):
    """
    Test AIS on 3x3 exponential-diagonal grid tensor network.
    
    Args:
        dim: Dimension of each tensor index
        A: Number of beta values in annealing schedule
        B: Number of parallel chains
        C: Number of iterations per beta step
        alpha: Exponential decay rate for diagonal entries
        show_diagnostics: Whether to show detailed output
    
    Returns:
        mean_Z: Mean estimate of partition function
        std_Z: Standard deviation of estimate
        rel_error: Relative error compared to exact result
    """
    burns = max(0, min(C // 10, C - 1))
    
    if show_diagnostics:
        print(f"\n[info] building 3x3 exp-diag tensor network (alpha={alpha})")
    
    G, tensors = build_3x3_grid_exp_diag(dim=dim, alpha=alpha)
    
    if show_diagnostics:
        print("[info] performing exact contraction...")
    TRUE_Z = contract_tensor_network(G, tensors)
    if show_diagnostics:
        print(f"[info] exact Z = {TRUE_Z:.12e}")

    tn = TensorNetwork(G, tensors)
    betas = make_logspace_betas(A)

    if show_diagnostics:
        print(f"\n[info] running AIS (A={A}, B={B}, C={C}, burns={burns})")
        print(f"[info] using logspace beta schedule")
    
    mean_Z, std_Z = run_multiple_chains(
        tn, betas,
        n_chains=B,
        iters=C,
        burns=burns,
        Z_true=TRUE_Z if show_diagnostics else None,
        verbose=show_diagnostics
    )

    rel_error = abs(mean_Z - TRUE_Z) / abs(TRUE_Z) if TRUE_Z != 0 else float('inf')
    
    if show_diagnostics:
        print("\n" + "=" * 60)
        print("[summary] 3x3 Exponential Diagonal Grid Results")
        print("=" * 60)
        print(f"[result] exact Z      : {TRUE_Z:.12e}")
        print(f"[result] estimated Z  : {mean_Z:.12e} ± {std_Z:.12e}")
        print(f"[result] rel error    : {rel_error:.6e}")
        print("=" * 60)

    return mean_Z, std_Z, rel_error


if __name__ == "__main__":
    test_trace_3x3_grid_exp_diag()
