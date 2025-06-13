# test on 2x2 ring tensor network: Tr(ABCD)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from src.algorithm import TensorNetwork, run_multiple_chains

def test_trace_ABCD(dim=3,
                    betas=np.linspace(0, 1, 200),
                    n_chains=5,
                    iters=20000,
                    burns=1900,
                    show_diagnostics=True):
    if show_diagnostics:
        print("\n>>> Building 2x2 ring tensor network (Tr(ABCD))")

    # create 2x2 ring tensor network
    G = nx.Graph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'),
        ('C', 'D'), ('D', 'A'),
    ])

    # init tensors with normal distribution for better numerical stability
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

    # exact contraction: trace of product
    A, B, C, D = [tensors[k][0] for k in ['A', 'B', 'C', 'D']]
    TRUE_Z = np.einsum('ij,jk,kl,li->', A, B, C, D)
    if show_diagnostics:
        print(f"True Z (Tr(ABCD)): {TRUE_Z:.6f}")

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
    test_trace_ABCD()