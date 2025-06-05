# test on 2x2 ring tensor network: Tr(ABCD)
import numpy as np
import networkx as nx
from ..src.algorithm import TensorNetwork, run_multiple_chains

def test_trace_ABCD(dim=3,
                    betas=np.linspace(0, 1, 100),
                    n_chains=5,
                    iters=10000,
                    burns=900,
                    n_rounds=5):
    print("\n>>> Building 2x2 ring tensor network (Tr(ABCD))")

    # create 2x2 ring tensor network
    G = nx.Graph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'),
        ('C', 'D'), ('D', 'A'),
    ])

    # positive tensors with shape (dim, dim)
    A, B, C, D = [np.random.rand(dim, dim) + 1e-6 for _ in range(4)]
    tensors = {
        'A': (A, ['i', 'j']),
        'B': (B, ['j', 'k']),
        'C': (C, ['k', 'l']),
        'D': (D, ['l', 'i'])
    }

    # exact contraction: trace of product
    TRUE_Z = np.einsum('ij,jk,kl,li->', A, B, C, D)
    print(f"True Z (Tr(ABCD)): {TRUE_Z:.6f}")

    tn = TensorNetwork(G, tensors)

    print("\n>>> Running MCMC chains")
    mean_Z, std_Z = run_multiple_chains(
        tn, betas,
        n_chains=n_chains,
        iters=iters,
        burns=burns,
        n_rounds=n_rounds,
        Z_true=TRUE_Z
    )

    rel_error = abs(mean_Z - TRUE_Z) / abs(TRUE_Z)
    print("\n================== Final Summary ==================")
    print(f"True       Z : {TRUE_Z:.6f}")
    print(f"Estimated  Z : {mean_Z:.6f} ± {std_Z:.6f}")
    print(f"Relative Error: {rel_error:.2e}")
    print("====================================================")

test_trace_ABCD()