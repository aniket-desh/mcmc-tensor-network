# Test AIS on 3x3 grid tensor network with diagonally dominant tensors
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


def build_3x3_grid_diagonally_dominant(dim=3, noise_level=0.1):
    """
    Build a 3x3 grid tensor network with diagonally dominant tensors.
    
    Args:
        dim: Dimension of each tensor index
        noise_level: Amount of noise to add to diagonal dominance (0 = pure diagonal, 1 = uniform)
    
    Returns:
        G: NetworkX graph representing the tensor network
        tensors: Dictionary mapping node names to (tensor, indices) tuples
    """
    np.random.seed(42)
    G = nx.Graph()
    tensors = {}
    grid_size = 3
    node_names = {(i, j): f"T{i}{j}" for i in range(grid_size) for j in range(grid_size)}

    def edge_index(i1, j1, i2, j2):
        return "_".join(sorted([f"{i1}{j1}", f"{i2}{j2}"]))

    for i in range(grid_size):
        for j in range(grid_size):
            name = node_names[(i, j)]
            
            physical_neighbors = []
            if i > 0: physical_neighbors.append((i - 1, j))
            if i < grid_size - 1: physical_neighbors.append((i + 1, j))
            if j > 0: physical_neighbors.append((i, j - 1))
            if j < grid_size - 1: physical_neighbors.append((i, j + 1))

            indices = sorted([edge_index(i, j, ni, nj) for ni, nj in physical_neighbors])
            G.add_node(name)
            for ni, nj in physical_neighbors:
                G.add_edge(name, node_names[(ni, nj)])
            
            shape = (dim,) * len(indices)
            rank = len(shape)

            # Create diagonal-dominant tensor
            diagonal_part = np.zeros(shape)
            for k in range(dim):
                idx = (k,) * rank 
                diagonal_part[idx] = 1.0

            noise_part = np.ones(shape)
            
            dominance_factor = 1.0 - noise_level
            tensor = diagonal_part * dominance_factor + noise_part * (noise_level / dim**rank)

            tensors[name] = (tensor, indices)

    return G, tensors


def make_logspace_betas(A):
    """Generate logspace beta schedule for better early-beta resolution."""
    betas = 1.0 - np.logspace(0, np.log10(1e-6), A)
    betas[0] = 0.0
    betas[-1] = 1.0
    return np.sort(betas)


def test_trace_3x3_grid_dd(dim=3,
                           A=200,
                           B=400,
                           C=200,
                           noise_level=0.1,
                           show_diagnostics=True):
    """
    Test AIS on 3x3 diagonally-dominant grid tensor network.
    
    Args:
        dim: Dimension of each tensor index
        A: Number of beta values in annealing schedule
        B: Number of parallel chains
        C: Number of iterations per beta step
        noise_level: Noise level for diagonal dominance
        show_diagnostics: Whether to show detailed output
    
    Returns:
        mean_Z: Mean estimate of partition function
        std_Z: Standard deviation of estimate
        rel_error: Relative error compared to exact result
    """
    burns = max(0, min(C // 10, C - 1))
    
    if show_diagnostics:
        print(f"\n[info] building 3x3 diagonally-dominant tensor network (noise_level={noise_level})")
    
    G, tensors = build_3x3_grid_diagonally_dominant(dim=dim, noise_level=noise_level)
    
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
        print("[summary] 3x3 Diagonally Dominant Grid Results")
        print("=" * 60)
        print(f"[result] exact Z      : {TRUE_Z:.12e}")
        print(f"[result] estimated Z  : {mean_Z:.12e} ± {std_Z:.12e}")
        print(f"[result] rel error    : {rel_error:.6e}")
        print("=" * 60)

    return mean_Z, std_Z, rel_error


if __name__ == "__main__":
    test_trace_3x3_grid_dd()
