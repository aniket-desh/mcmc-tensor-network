import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.algorithm import TensorNetwork, run_multiple_chains

def test_star_network(dim=2,  # Binary qubits (dim=2)
                     betas=np.linspace(0, 1, 200),
                     n_chains=5,
                     iters=20000,
                     burns=1900,
                     show_diagnostics=True):
    if show_diagnostics:
        print("\n>>> Building 5-qubit star tensor network")

    # Create star-shaped graph (central tensor + 4 peripheral tensors)
    G = nx.Graph()
    G.add_edges_from([
        ('C', 'P1'), ('C', 'P2'), ('C', 'P3'), ('C', 'P4')
    ])

    # Initialize tensors (random normal for stability)
    tensors = {}
    index_order = {
        'C': ['a', 'b', 'c', 'd'],  # Central tensor (4 indices)
        'P1': ['a', 'e'],           # Peripheral tensors (connected to center)
        'P2': ['b', 'f'],
        'P3': ['c', 'g'],
        'P4': ['d', 'h']
    }

    for node, inds in index_order.items():
        shape = (dim,) * len(inds)
        data = np.random.normal(loc=1.0, scale=0.1, size=shape) + 1e-6
        tensors[node] = (data, inds)

    # Exact contraction using np.einsum
    C = tensors['C'][0]
    P1, P2, P3, P4 = [tensors[f'P{i}'][0] for i in range(1, 5)]
    TRUE_Z = np.einsum('abcd,ae,bf,cg,dh->efgh', C, P1, P2, P3, P4).sum()
    if show_diagnostics:
        print(f"True Z (5-qubit star): {TRUE_Z:.6f}")

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
    test_star_network()