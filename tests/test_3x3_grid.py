# Test AIS on 3x3 grid tensor network with Gaussian entries

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def build_3x3_grid_gaussian(dim=3, seed=42):
    """
    Build a 3x3 grid tensor network with Gaussian entries.
    
    Args:
        dim: Dimension of each tensor index
        seed: Random seed for reproducibility
    
    Returns:
        G: NetworkX graph representing the tensor network
        tensors: Dictionary mapping node names to (tensor, indices) tuples
    """
    np.random.seed(seed)
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
            data = np.random.normal(loc=1.0, scale=0.1, size=shape) + 1e-6
            tensors[name] = (data, neighbors)

    return G, tensors


def make_logspace_betas(A):
    """Generate logspace beta schedule for better early-beta resolution."""
    betas = 1.0 - np.logspace(0, np.log10(1e-6), A)
    betas[0] = 0.0
    betas[-1] = 1.0
    return np.sort(betas)


def test_trace_3x3_grid(dim=3,
                        A=200,
                        B=400,
                        C=200,
                        seed=42,
                        show_diagnostics=True):
    """
    Test AIS on 3x3 Gaussian grid tensor network.
    
    Args:
        dim: Dimension of each tensor index
        A: Number of beta values in annealing schedule
        B: Number of parallel chains
        C: Number of iterations per beta step
        seed: Random seed for reproducibility
        show_diagnostics: Whether to show detailed output
    
    Returns:
        mean_Z: Mean estimate of partition function
        std_Z: Standard deviation of estimate
        rel_error: Relative error compared to exact result
    """
    burns = max(0, min(C // 10, C - 1))
    
    if show_diagnostics:
        print(f"\n[info] building 3x3 Gaussian tensor network")
    
    G, tensors = build_3x3_grid_gaussian(dim=dim, seed=seed)
    
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
        print("[summary] 3x3 Gaussian Grid Results")
        print("=" * 60)
        print(f"[result] exact Z      : {TRUE_Z:.12e}")
        print(f"[result] estimated Z  : {mean_Z:.12e} ± {std_Z:.12e}")
        print(f"[result] rel error    : {rel_error:.6e}")
        print("=" * 60)

    return mean_Z, std_Z, rel_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AIS on 3x3 Gaussian grid tensor network")
    parser.add_argument("--dim", type=int, default=3, help="Bond dimension")
    parser.add_argument("-A", type=int, default=200, help="Number of beta values")
    parser.add_argument("-B", type=int, default=400, help="Number of parallel chains")
    parser.add_argument("-C", type=int, default=200, help="Iterations per beta step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    test_trace_3x3_grid(
        dim=args.dim,
        A=args.A,
        B=args.B,
        C=args.C,
        seed=args.seed,
    )
