# Test AIS on 2x2 ring tensor network: Tr(ABCD)

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import networkx as nx
from src.algorithm import TensorNetwork, run_multiple_chains


def make_logspace_betas(A):
    """Generate logspace beta schedule for better early-beta resolution."""
    betas = 1.0 - np.logspace(0, np.log10(1e-6), A)
    betas[0] = 0.0
    betas[-1] = 1.0
    return np.sort(betas)


def test_trace_ABCD(dim=3,
                    A=200,
                    B=100,
                    C=200,
                    seed=42,
                    show_diagnostics=True):
    """
    Test AIS on 2x2 ring tensor network (Tr(ABCD)).
    
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
        print(f"\n[info] building 2x2 ring tensor network (Tr(ABCD))")

    # Create 2x2 ring tensor network
    G = nx.Graph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'),
        ('C', 'D'), ('D', 'A'),
    ])

    # Initialize tensors with normal distribution
    np.random.seed(seed)
    tensors = {}
    index_order = {
        'A': ['i', 'j'],
        'B': ['j', 'k'],
        'C': ['k', 'l'],
        'D': ['l', 'i']
    }
    for node, inds in index_order.items():
        data = np.random.normal(loc=1.0, scale=0.1, size=(dim, dim)) + 1e-6
        tensors[node] = (data, inds)

    # Exact contraction: trace of product
    A_t, B_t, C_t, D_t = [tensors[k][0] for k in ['A', 'B', 'C', 'D']]
    TRUE_Z = np.einsum('ij,jk,kl,li->', A_t, B_t, C_t, D_t)
    
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
        print("[summary] 2x2 Ring (Tr(ABCD)) Results")
        print("=" * 60)
        print(f"[result] exact Z      : {TRUE_Z:.12e}")
        print(f"[result] estimated Z  : {mean_Z:.12e} ± {std_Z:.12e}")
        print(f"[result] rel error    : {rel_error:.6e}")
        print("=" * 60)

    return mean_Z, std_Z, rel_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AIS on 2x2 ring tensor network (Tr(ABCD))")
    parser.add_argument("--dim", type=int, default=3, help="Bond dimension")
    parser.add_argument("-A", type=int, default=200, help="Number of beta values")
    parser.add_argument("-B", type=int, default=100, help="Number of parallel chains")
    parser.add_argument("-C", type=int, default=200, help="Iterations per beta step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    test_trace_ABCD(
        dim=args.dim,
        A=args.A,
        B=args.B,
        C=args.C,
        seed=args.seed,
    )
