# Test AIS on 3x3 grid tensor network with uniform-around-1 entries

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


def build_3x3_grid_uniform1(dim=3, jitter=0.1, eps=1e-6, seed=42):
    """
    Build a 3x3 grid tensor network with uniform entries around 1.
    
    Each tensor entry is drawn from U[1-jitter, 1+jitter], clipped to be >= eps.
    
    Args:
        dim: Dimension of each tensor index
        jitter: Half-width of uniform distribution around 1
        eps: Minimum value floor
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
        return "_".join(sorted([f"{i1}{j1}", f"{i2}{j2}"]))

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
            
            # Uniform around 1
            T = np.random.uniform(1.0 - jitter, 1.0 + jitter, size=shape)
            T = np.maximum(T, eps)
            
            tensors[name] = (T, indices)

    return G, tensors


def make_logspace_betas(A):
    """Generate logspace beta schedule for better early-beta resolution."""
    betas = 1.0 - np.logspace(0, np.log10(1e-6), A)
    betas[0] = 0.0
    betas[-1] = 1.0
    return np.sort(betas)


def test_trace_3x3_grid_uniform1(dim=3,
                                  A=200,
                                  B=400,
                                  C=200,
                                  jitter=0.1,
                                  seed=42,
                                  show_diagnostics=True):
    """
    Test AIS on 3x3 uniform-around-1 grid tensor network.
    
    Args:
        dim: Dimension of each tensor index
        A: Number of beta values in annealing schedule
        B: Number of parallel chains
        C: Number of iterations per beta step
        jitter: Half-width of uniform distribution around 1
        seed: Random seed for reproducibility
        show_diagnostics: Whether to show detailed output
    
    Returns:
        mean_Z: Mean estimate of partition function
        std_Z: Standard deviation of estimate
        rel_error: Relative error compared to exact result
    """
    burns = max(0, min(C // 10, C - 1))
    
    if show_diagnostics:
        print(f"\n[info] building 3x3 uniform1 tensor network (jitter={jitter})")
    
    G, tensors = build_3x3_grid_uniform1(dim=dim, jitter=jitter, seed=seed)
    
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
        print("[summary] 3x3 Uniform1 Grid Results")
        print("=" * 60)
        print(f"[result] exact Z      : {TRUE_Z:.12e}")
        print(f"[result] estimated Z  : {mean_Z:.12e} ± {std_Z:.12e}")
        print(f"[result] rel error    : {rel_error:.6e}")
        print("=" * 60)

    return mean_Z, std_Z, rel_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AIS on 3x3 uniform-around-1 grid tensor network")
    parser.add_argument("--dim", type=int, default=3, help="Bond dimension")
    parser.add_argument("-A", type=int, default=200, help="Number of beta values")
    parser.add_argument("-B", type=int, default=400, help="Number of parallel chains")
    parser.add_argument("-C", type=int, default=200, help="Iterations per beta step")
    parser.add_argument("--jitter", type=float, default=0.1, help="Uniform jitter half-width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    test_trace_3x3_grid_uniform1(
        dim=args.dim,
        A=args.A,
        B=args.B,
        C=args.C,
        jitter=args.jitter,
        seed=args.seed,
    )

