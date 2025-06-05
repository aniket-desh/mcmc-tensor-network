import numpy as np
import networkx as nx
from ..src.algorithm import TensorNetwork, run_multiple_chains

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

# full test on 3x3 grid tensor network with summary
def test_trace_3x3_grid(dim=3,
                        betas=np.linspace(0, 1, 100),
                        n_chains=5,
                        iters=20000,
                        burns=2000,
                        n_rounds=50):
    print("\n>>> Building 3x3 tensor network")
    G, tensors = build_3x3_grid_test(dim=dim)
    TRUE_Z = contract_tensor_network(G, tensors)
    print(f"True Z (exact contraction): {TRUE_Z:.6f}")

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

    rel_error = abs(mean_Z - TRUE_Z) / TRUE_Z
    print("\n================== Final Summary ==================")
    print(f"True       Z : {TRUE_Z:.6f}")
    print(f"Estimated  Z : {mean_Z:.6f} ± {std_Z:.6f}")
    print(f"Relative Error: {rel_error:.2e}")
    print("====================================================")

test_trace_3x3_grid()