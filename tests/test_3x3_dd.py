import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from src.algorithm import TensorNetwork, run_multiple_chains

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
                    raise ValueError("ran out of characters for einsum indices")
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

            indices = sorted([edge_index(i,j, ni,nj) for ni,nj in physical_neighbors])
            G.add_node(name)
            for ni, nj in physical_neighbors:
                 G.add_edge(name, node_names[(ni,nj)])
            
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

def test_trace_3x3_grid_dd(dim=3,
                           n_betas=200,
                           n_chains=10,
                           iters=20000,
                           burns=10000,
                           show_diagnostics=True):
    """
    Test AIS on 3x3 diagonally-dominant grid tensor network.
    
    Args:
        dim: Dimension of each tensor index
        n_betas: Number of beta values in annealing schedule
        n_chains: Number of parallel chains
        iters: Number of iterations per beta step
        burns: Number of burn-in iterations per beta step
        show_diagnostics: Whether to show diagnostic plots and detailed output
    
    Returns:
        mean_Z: Mean estimate of partition function
        std_Z: Standard deviation of estimate
        rel_error: Relative error compared to exact result
    """
    if show_diagnostics:
        print("\n>>> Building 3x3 diagonally-dominant tensor network")
    
    G, tensors = build_3x3_grid_diagonally_dominant(dim=dim, noise_level=0.1)
    
    if show_diagnostics:
        print(">>> Performing exact contraction (this may take a moment)...")
    TRUE_Z = contract_tensor_network(G, tensors)
    if show_diagnostics:
        print(f"True Z (exact contraction): {TRUE_Z:.12f}")

    tn = TensorNetwork(G, tensors)
    betas = np.linspace(0, 1, n_betas)**2  # Quadratic beta schedule

    if show_diagnostics:
        print("\n>>> Running AIS estimation...")
    mean_Z, std_Z = run_multiple_chains(
        tn, betas,
        n_chains=n_chains,
        iters=iters,
        burns=burns,
        Z_true=TRUE_Z if show_diagnostics else None
    )

    rel_error = abs(mean_Z - TRUE_Z) / abs(TRUE_Z) if TRUE_Z != 0 else float('inf')
    
    if show_diagnostics:
        print("\n================== Final Summary ==================")
        print(f"True        Z  : {TRUE_Z:.12f}")
        print(f"Estimated   Z  : {mean_Z:.12f} ± {std_Z:.12f}")
        print(f"Relative Error : {rel_error:.6e}")
        print("====================================================")

    return mean_Z, std_Z, rel_error

if __name__ == "__main__":
    test_trace_3x3_grid_dd()
