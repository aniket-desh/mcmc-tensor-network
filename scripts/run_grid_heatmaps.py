#!/usr/bin/env python3
"""
Cost optimization experiments for 3x3 grid tensor networks.
Generates heatmaps showing relative error across different (A, B, C) budget allocations.
"""

from __future__ import annotations
import sys
from pathlib import Path
import argparse

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd
import time
from functools import partial

from src.algorithm import (
    TensorNetwork,
    run_multiple_chains,
    contract_tensor_network
)


# ============================================================================
# Helper functions for geometric statistics on relative errors
# ============================================================================

EPS = 1e-12

def gmean_series(s):
    """Geometric mean of a pandas Series."""
    x = np.clip(s.to_numpy(np.float64), EPS, None)
    if len(x) == 0 or np.any(~np.isfinite(x)):
        return np.nan
    log_x = np.log(x)
    if not np.all(np.isfinite(log_x)):
        return np.nan
    return float(np.exp(np.mean(log_x)))


def gvar_series(s):
    """Variance of log(x) for a pandas Series."""
    x = np.clip(s.to_numpy(np.float64), EPS, None)
    if len(x) < 2 or np.any(~np.isfinite(x)):
        return np.nan
    log_x = np.log(x)
    if not np.all(np.isfinite(log_x)):
        return np.nan
    if np.allclose(log_x, log_x[0]):
        return 0.0
    return float(np.var(log_x, ddof=1))


def gstd_series(s):
    """
    Compute true geometric standard deviation: GSD(ε) = exp(Std(log ε))
    Returns ≥ 1, where 1 means no multiplicative spread.
    """
    x = np.clip(s.to_numpy(np.float64), EPS, None)
    if len(x) < 2 or np.any(~np.isfinite(x)):
        return np.nan
    log_x = np.log(x)
    if not np.all(np.isfinite(log_x)):
        return np.nan
    std_log = np.std(log_x, ddof=1)
    if np.allclose(std_log, 0.0):
        return 1.0
    return float(np.exp(std_log))


# ============================================================================
# Tensor network creation
# ============================================================================

def create_test_tensor_network(network_type="2x2_ring", dim=3, seed=42,
                               eps=1e-6, alpha=0.6, spike_factor=10.0, jitter=0.1):
    """
    network_type options (topology + optional weight model):
      - "2x2_ring", "3x3_grid"                      (old behavior: N(1,0.1))
      - "2x2_ring_uniform1",   "3x3_grid_uniform1"  (uniform around 1, nonnegative)
      - "2x2_ring_diagexp",    "3x3_grid_diagexp"   (exponentially-decaying diagonal)
      - "2x2_ring_spikes",     "3x3_grid_spikes"    (uniform + diagonal spikes)

    dim: index range size for each leg
    alpha: decay rate for diagexp
    spike_factor: multiplicative spike on diagonal entries for spikes
    jitter: half-width of uniform around 1, i.e., U[1-jitter, 1+jitter]
    """
    np.random.seed(seed)

    if "_" in network_type:
        topo, model = network_type.rsplit("_", 1)
    else:
        topo, model = network_type, "gauss"

    def sample_tensor(shape, rng=np.random):
        k = len(shape)

        if model == "gauss":
            T = rng.normal(loc=1.0, scale=0.1, size=shape) + eps
            return T

        if model == "uniform1":
            T = rng.uniform(1.0 - jitter, 1.0 + jitter, size=shape)
            T = np.maximum(T, eps)
            return T

        if model == "diagexp":
            idx = np.indices(shape)
            mask = np.all(idx == idx[0], axis=0)
            T = np.full(shape, eps, dtype=float)
            diag_vals = np.exp(-alpha * np.arange(shape[0]))
            T[mask] = diag_vals[idx[0][mask]]
            return T

        if model == "spikes":
            T = rng.uniform(1.0 - jitter, 1.0 + jitter, size=shape)
            T = np.maximum(T, eps)
            idx = np.indices(shape)
            mask = np.all(idx == idx[0], axis=0)
            T[mask] *= spike_factor
            return T

        raise ValueError(f"Unknown weight model in network_type: {model}")

    if topo == "2x2_ring":
        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])

        tensors = {}
        index_order = {'A': ['i', 'j'], 'B': ['j', 'k'], 'C': ['k', 'l'], 'D': ['l', 'i']}

        for node, inds in index_order.items():
            data = sample_tensor((dim, dim))
            tensors[node] = (data, inds)

        A_, B_, C_, D_ = [tensors[k][0] for k in ['A', 'B', 'C', 'D']]
        Z_true = np.einsum('ij,jk,kl,li->', A_, B_, C_, D_)

    elif topo == "3x3_grid":
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
                    neighbors.append(edge_index(i, j, i, j+1))
                    G.add_edge(name, node_names[(i, j+1)])
                if i + 1 < grid_size:
                    neighbors.append(edge_index(i, j, i+1, j))
                    G.add_edge(name, node_names[(i+1, j)])
                if j > 0:
                    neighbors.append(edge_index(i, j-1, i, j))
                if i > 0:
                    neighbors.append(edge_index(i-1, j, i, j))

                neighbors = sorted(neighbors)
                shape = (dim,) * len(neighbors)
                data = sample_tensor(shape)
                tensors[name] = (data, neighbors)

        Z_true = None

    else:
        raise ValueError(f"Unknown topology in network_type: {network_type}")

    tn = TensorNetwork(G, tensors)
    return tn, Z_true


# ============================================================================
# Experiment runner
# ============================================================================

def run_ais_experiment(tn, A, B, C, seed=None, verbose=False, return_time=True):
    """Run single AIS experiment with given budget allocation."""
    if seed is not None:
        np.random.seed(int(seed))

    betas = np.linspace(0, 1, A)
    betas[0] = 0.0
    betas[-1] = 1.0

    start_time = time.time() if return_time else None

    Z_est, Z_std = run_multiple_chains(
        tn, betas,
        iters=C,
        burns=min(C//10, C-1),
        n_chains=B,
        verbose=verbose,
    )

    end_time = time.time() if return_time else None
    wall_time = (end_time - start_time) if return_time else None

    return {
        'Z_est': Z_est,
        'Z_std': Z_std,
        'time_seconds': wall_time,
        'A': A, 'B': B, 'C': C,
        'N': A * B * C,
        'seed': seed
    }


def compute_relative_error(Z_est, Z_true):
    """Compute relative error between estimate and ground truth."""
    if Z_true is None:
        return np.nan
    return abs(Z_est - Z_true) / abs(Z_true)


def generate_cost_experiment_data(
    tn,
    N_values,
    A_range,
    B_range,
    Z_true=None,
    n_trials=10,
    seeds=None,
    reseed_network=False,
    network_factory=None,
    verbose=True
):
    """
    Generate cost optimization experiment data across parameter grid.

    If reseed_network is True, you must pass network_factory, a callable (seed -> (tn, Z_true_opt)).
    """
    results = []

    if seeds is None:
        seeds = list(range(n_trials))

    total = len(N_values) * len(A_range) * len(B_range) * len(seeds)
    count = 0

    base_tn = tn
    base_Z_true = Z_true

    for N in N_values:
        for A in A_range:
            for B in B_range:
                if N % (A * B) != 0:
                    continue
                C = N // (A * B)
                if C < 1:
                    continue

                for seed in seeds:
                    count += 1
                    if verbose and count % 10 == 0:
                        print(f"[info] progress: {count}/{total} | N={N}, A={A}, B={B}, C={C}, seed={seed}")

                    if reseed_network:
                        if network_factory is None:
                            raise ValueError("reseed_network=True requires network_factory.")
                        tn_used, Z_true_seed = network_factory(seed=seed)
                        Z_true_used = Z_true_seed if Z_true_seed is not None else base_Z_true
                    else:
                        tn_used = base_tn
                        Z_true_used = base_Z_true

                    out = run_ais_experiment(tn_used, A, B, C, seed=seed, verbose=False)
                    rel_err = compute_relative_error(out['Z_est'], Z_true_used)

                    results.append({
                        'N': N, 'A': A, 'B': B, 'C': C, 'seed': seed,
                        'Z_est': out['Z_est'],
                        'Z_std': out['Z_std'],
                        'rel_error': rel_err,
                        'time_seconds': out['time_seconds']
                    })

    return pd.DataFrame(results)


# ============================================================================
# Plotting functions
# ============================================================================

def plot_cost_heatmap(df, N_value, metric='rel_error', aggregation='gmean', save_path=None):
    """Plot dual heatmap showing metric and C values."""
    data = df[df['N'] == N_value].copy()

    if data.empty:
        print(f"[error] no data for N={N_value}")
        return

    if aggregation == 'mean':
        agg_data = data.groupby(['A', 'B'])[metric].mean().reset_index()
    elif aggregation == 'median':
        agg_data = data.groupby(['A', 'B'])[metric].median().reset_index()
    elif aggregation == 'min':
        agg_data = data.groupby(['A', 'B'])[metric].min().reset_index()
    elif aggregation == 'gmean':
        agg = data.groupby(['A','B'])[metric].apply(gmean_series)
        agg_data = agg.reset_index(name=metric)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    pivot = agg_data.pivot(index='A', columns='B', values=metric)
    C_pivot = data.groupby(['A', 'B'])['C'].first().reset_index().pivot(index='A', columns='B', values='C')
    C_pivot = C_pivot.fillna(0).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(pivot, annot=True, fmt='.2e' if metric == 'rel_error' else '.2f',
                cmap='viridis', ax=ax1)
    ax1.set_title(f'{metric.replace("_", " ").title()} ({aggregation})\nN = {N_value}')
    ax1.set_xlabel('B (number of chains)')
    ax1.set_ylabel('A (beta ladder length)')

    sns.heatmap(C_pivot, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title(f'Corresponding C values\n(iterations per chain per beta)')
    ax2.set_xlabel('B (number of chains)')
    ax2.set_ylabel('A (beta ladder length)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] {save_path}")
    else:
        plt.show()


def plot_variance_heatmap(df, N_value, metric='rel_error', mode='logvar', save_path=None):
    """Plot dual heatmap showing variance/gstd and C values."""
    data = df[df['N'] == N_value].copy()
    if data.empty:
        print(f"[error] no data for N={N_value}")
        return

    if mode == 'logvar':
        agg = data.groupby(['A','B'])[metric].apply(gvar_series).reset_index(name='logvar')
        values = agg.pivot(index='A', columns='B', values='logvar')
        title = 'Variance of log(Rel Error)'
        fmt = '.2e'
        cmap = 'magma'
    elif mode == 'gstd':
        agg = data.groupby(['A','B'])[metric].apply(gstd_series).reset_index(name='gstd')
        values = agg.pivot(index='A', columns='B', values='gstd')
        title = 'Geometric Std Dev of Rel Error'
        fmt = '.2f'
        cmap = 'magma_r'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    C_pivot = data.groupby(['A', 'B'])['C'].first().reset_index().pivot(index='A', columns='B', values='C')
    C_pivot = C_pivot.fillna(0).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(values, annot=True, fmt=fmt, cmap=cmap, ax=ax1)
    ax1.set_title(f'{title}\nN = {N_value}')
    ax1.set_xlabel('B (number of chains)')
    ax1.set_ylabel('A (beta ladder length)')

    sns.heatmap(C_pivot, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Corresponding C values\n(iterations per chain per beta)')
    ax2.set_xlabel('B (number of chains)')
    ax2.set_ylabel('A (beta ladder length)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] {save_path}")
    else:
        plt.show()


# ============================================================================
# Main execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cost optimization experiments for 3x3 grid tensor networks"
    )

    # Network parameters
    parser.add_argument("--network-type", type=str, default="3x3_grid_diagexp",
                       help="Network type (e.g., 3x3_grid, 3x3_grid_diagexp, 3x3_grid_spikes, 3x3_grid_uniform1)")
    parser.add_argument("--dim", type=int, default=4,
                       help="Bond dimension")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for network generation")

    # Budget parameters
    parser.add_argument("--N", type=int, nargs='+', default=[100000],
                       help="Total budget N (can specify multiple)")
    parser.add_argument("--A-range", type=int, nargs='+', default=[20, 50, 100, 200, 500],
                       help="Beta ladder lengths to test")
    parser.add_argument("--B-range", type=int, nargs='+', default=[5, 10, 20, 25, 50, 100, 200],
                       help="Number of chains to test")

    # Trial parameters
    parser.add_argument("--seeds", type=int, nargs='+', default=[11, 23, 42, 77, 101],
                       help="Random seeds for trials")

    # Output parameters
    parser.add_argument("--output", type=str, default="cost_optimization_grid.csv",
                       help="Output CSV file")
    parser.add_argument("--plots-dir", type=str, default=None,
                       help="Directory to save plots (if not specified, plots are displayed)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")

    args = parser.parse_args()

    # Create tensor network
    print(f"[info] building {args.network_type} tensor network (dim={args.dim}, seed={args.seed})")
    tn, Z_true = create_test_tensor_network(args.network_type, dim=args.dim, seed=args.seed)

    if Z_true is None:
        print('[info] computing reference Z_true via exact contraction...')
        Z_true = contract_tensor_network(tn.graph, tn.tensors)

    print(f"[info] true contraction Z = {Z_true:.6e}")

    # Display parameter ranges
    print(f"\n[info] parameter ranges:")
    print(f"  N = {args.N}")
    print(f"  A (beta steps) = {args.A_range}")
    print(f"  B (chains) = {args.B_range}")
    print(f"  seeds = {args.seeds}")

    # Show valid combinations
    print(f"\n[info] valid (A,B,C) combinations for N={args.N[0]}:")
    valid_combinations = []
    for A in args.A_range:
        for B in args.B_range:
            if args.N[0] % (A * B) == 0:
                C = args.N[0] // (A * B)
                if C >= 1:
                    valid_combinations.append((A, B, C))
                    print(f"  A={A:3d}, B={B:3d}, C={C:4d}")

    num_trials = len(args.seeds)
    total_experiments = len(valid_combinations) * num_trials
    print(f"\n[info] total valid combinations: {len(valid_combinations)}")
    print(f"[info] running {len(valid_combinations)} combinations x {num_trials} trials = {total_experiments} total experiments")

    # Run experiments
    start_time = time.time()
    results_df = generate_cost_experiment_data(
        tn=tn,
        N_values=args.N,
        A_range=args.A_range,
        B_range=args.B_range,
        Z_true=Z_true,
        seeds=args.seeds,
        verbose=True
    )
    end_time = time.time()

    print(f"\n[done] experiments completed in {end_time - start_time:.1f} seconds")
    print(f"[info] generated {len(results_df)} data points")

    # Summary statistics
    print(f"\n[summary] stats:")
    print(f"  rel error range: {results_df['rel_error'].min():.2e} to {results_df['rel_error'].max():.2e}")
    overall_gm = gmean_series(results_df['rel_error'].dropna())
    print(f"  geom mean rel error: {overall_gm:.2e}")
    print(f"  time per experiment: {results_df['time_seconds'].mean():.3f} ± {results_df['time_seconds'].std():.3f}s")

    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"[done] saved -> {args.output}")

    # Generate plots
    if not args.no_plots:
        for N_val in args.N:
            # Geometric mean relative error
            save_path = f"{args.plots_dir}/grid_gmean_rel_error_N{N_val}.png" if args.plots_dir else None
            plot_cost_heatmap(results_df, N_value=N_val, metric='rel_error',
                            aggregation='gmean', save_path=save_path)

            # Variance of log relative error
            save_path = f"{args.plots_dir}/grid_var_log_rel_error_N{N_val}.png" if args.plots_dir else None
            plot_variance_heatmap(results_df, N_value=N_val, metric='rel_error',
                                mode='logvar', save_path=save_path)

            # Geometric std dev
            save_path = f"{args.plots_dir}/grid_gstd_rel_error_N{N_val}.png" if args.plots_dir else None
            plot_variance_heatmap(results_df, N_value=N_val, metric='rel_error',
                                mode='gstd', save_path=save_path)


if __name__ == "__main__":
    main()
