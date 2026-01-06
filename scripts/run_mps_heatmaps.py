#!/usr/bin/env python3
"""
Cost optimization experiments for periodic MPS tensor networks.
Generates heatmaps showing relative error across different (A, B, C) budget allocations
for multiple tensor types (diagexp, spikes, uniform1).
"""

from __future__ import annotations
import sys
from pathlib import Path
import argparse

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

from src.algorithm import (
    TensorNetwork,
    run_multiple_chains,
)

from tests.test_periodic_mps import (
    build_periodic_mps,
    exact_contract_periodic_mps,
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
# MPS tensor network creation
# ============================================================================

def create_mps_tensor_network(tensor_type="gaussian", n_sites=16, dim=3, seed=42,
                               eps=1e-6, alpha=0.6, spike_factor=10.0, jitter=0.1):
    """
    Create a periodic MPS tensor network with specified tensor type.

    tensor_type options:
      - "gaussian"   (N(1, 0.1), clipped >= eps)
      - "uniform1"   (U[1-jitter, 1+jitter], clipped >= eps)
      - "diagexp"    (diagonal exp(-alpha*k), off-diagonal eps)
      - "spikes"     (uniform1 + diagonal * spike_factor)
    """
    G, tensors = build_periodic_mps(
        n_sites=n_sites,
        dim=dim,
        tensor_type=tensor_type,
        seed=seed,
        eps=eps,
        alpha=alpha,
        spike_factor=spike_factor,
        jitter=jitter,
    )

    Z_true = exact_contract_periodic_mps(tensors, n_sites, dim)
    tn = TensorNetwork(G, tensors)

    return tn, Z_true


# ============================================================================
# Experiment runner
# ============================================================================

def run_ais_experiment(tn, A, B, C, seed=None, verbose=False, return_time=True):
    """Run single AIS experiment with given budget allocation."""
    if seed is not None:
        np.random.seed(int(seed))

    # logspace beta schedule
    betas = 1.0 - np.logspace(0, np.log10(1e-6), A)
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


def generate_mps_experiment_data(
    tensor_type,
    n_sites,
    dim,
    N_values,
    A_range,
    B_range,
    seeds,
    eps=1e-6,
    alpha=0.6,
    spike_factor=10.0,
    jitter=0.1,
    verbose=True
):
    """
    Generate experiment data for periodic MPS with specified tensor type.
    """
    results = []
    total = len(N_values) * len(A_range) * len(B_range) * len(seeds)
    count = 0

    # create the base tensor network (fixed seed for network structure)
    tn, Z_true = create_mps_tensor_network(
        tensor_type=tensor_type,
        n_sites=n_sites,
        dim=dim,
        seed=42,  # fixed seed for network
        eps=eps,
        alpha=alpha,
        spike_factor=spike_factor,
        jitter=jitter,
    )

    if verbose:
        print(f"[info] tensor type: {tensor_type}, n_sites={n_sites}, dim={dim}")
        print(f"[info] Z_true = {Z_true:.6e}")

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
                        print(f"[info] progress: {count}/{total} | A={A}, B={B}, C={C}, seed={seed}")

                    out = run_ais_experiment(tn, A, B, C, seed=seed, verbose=False)
                    rel_err = compute_relative_error(out['Z_est'], Z_true)

                    results.append({
                        'tensor_type': tensor_type,
                        'n_sites': n_sites,
                        'dim': dim,
                        'N': N, 'A': A, 'B': B, 'C': C, 'seed': seed,
                        'Z_est': out['Z_est'],
                        'Z_std': out['Z_std'],
                        'Z_true': Z_true,
                        'rel_error': rel_err,
                        'time_seconds': out['time_seconds']
                    })

    return pd.DataFrame(results)


# ============================================================================
# Plotting functions
# ============================================================================

def plot_cost_heatmap(df, N_value, tensor_type, metric='rel_error', aggregation='gmean', save_path=None):
    """Plot dual heatmap showing metric and C values."""
    data = df[(df['N'] == N_value) & (df['tensor_type'] == tensor_type)].copy()

    if data.empty:
        print(f"[error] no data for N={N_value}, tensor_type={tensor_type}")
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
    ax1.set_title(f'{tensor_type.upper()} - {metric.replace("_", " ").title()} ({aggregation})\nN = {N_value}')
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


def plot_variance_heatmap(df, N_value, tensor_type, metric='rel_error', mode='logvar', save_path=None):
    """Plot dual heatmap showing variance/gstd and C values."""
    data = df[(df['N'] == N_value) & (df['tensor_type'] == tensor_type)].copy()
    if data.empty:
        print(f"[error] no data for N={N_value}, tensor_type={tensor_type}")
        return

    if mode == 'logvar':
        agg = data.groupby(['A','B'])[metric].apply(gvar_series).reset_index(name='logvar')
        values = agg.pivot(index='A', columns='B', values='logvar')
        title = f'{tensor_type.upper()} - Variance of log(Rel Error)'
        fmt = '.2e'
        cmap = 'magma'
    elif mode == 'gstd':
        agg = data.groupby(['A','B'])[metric].apply(gstd_series).reset_index(name='gstd')
        values = agg.pivot(index='A', columns='B', values='gstd')
        title = f'{tensor_type.upper()} - Geometric Std Dev of Rel Error'
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
        description="Cost optimization experiments for periodic MPS tensor networks"
    )

    # MPS parameters
    parser.add_argument("--n-sites", type=int, default=16,
                       help="Number of sites in periodic MPS")
    parser.add_argument("--dim", type=int, default=3,
                       help="Bond dimension")
    parser.add_argument("--tensor-types", type=str, nargs='+', default=['diagexp', 'spikes', 'uniform1'],
                       help="Tensor types to test")

    # Model-specific parameters
    parser.add_argument("--alpha", type=float, default=0.6,
                       help="Decay rate for diagexp tensors")
    parser.add_argument("--spike-factor", type=float, default=10.0,
                       help="Spike multiplier for spikes tensors")
    parser.add_argument("--jitter", type=float, default=0.1,
                       help="Jitter for uniform1 tensors (U[1-jitter, 1+jitter])")
    parser.add_argument("--eps", type=float, default=1e-6,
                       help="Minimum tensor value")

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
    parser.add_argument("--output", type=str, default="cost_optimization_mps.csv",
                       help="Output CSV file")
    parser.add_argument("--plots-dir", type=str, default=None,
                       help="Directory to save plots (if not specified, plots are displayed)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")

    args = parser.parse_args()

    # Display configuration
    print(f"[config] Periodic MPS: n_sites={args.n_sites}, dim={args.dim}")
    print(f"[config] Tensor types: {args.tensor_types}")
    print(f"[config] N = {args.N}")
    print(f"[config] A (beta steps) = {args.A_range}")
    print(f"[config] B (chains) = {args.B_range}")
    print(f"[config] Seeds: {args.seeds}")

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

    total_experiments = len(valid_combinations) * len(args.seeds) * len(args.tensor_types)
    print(f"\n[info] {len(valid_combinations)} combinations x {len(args.seeds)} seeds x {len(args.tensor_types)} types = {total_experiments} total experiments")

    # Run experiments for all tensor types
    all_results = []
    start_time = time.time()

    for tensor_type in args.tensor_types:
        print(f"\n{'='*60}")
        print(f"[running] tensor_type = {tensor_type}")
        print(f"{'='*60}")

        df = generate_mps_experiment_data(
            tensor_type=tensor_type,
            n_sites=args.n_sites,
            dim=args.dim,
            N_values=args.N,
            A_range=args.A_range,
            B_range=args.B_range,
            seeds=args.seeds,
            alpha=args.alpha,
            spike_factor=args.spike_factor,
            jitter=args.jitter,
            eps=args.eps,
            verbose=True
        )
        all_results.append(df)

        # Quick stats
        print(f"\n[summary] {tensor_type}:")
        print(f"  rel_error range: {df['rel_error'].min():.2e} to {df['rel_error'].max():.2e}")
        print(f"  geom mean: {gmean_series(df['rel_error'].dropna()):.2e}")

    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)

    end_time = time.time()
    print(f"\n{'='*60}")
    print(f"[done] all experiments completed in {end_time - start_time:.1f} seconds")
    print(f"[info] total data points: {len(results_df)}")

    # Save combined results
    results_df.to_csv(args.output, index=False)
    print(f"[saved] {args.output}")

    # Detailed summary
    print(f"\n{'='*70}")
    print("SUMMARY: Periodic MPS Cost Analysis")
    print(f"n_sites={args.n_sites}, dim={args.dim}, N={args.N[0]}")
    print(f"{'='*70}")

    for tensor_type in args.tensor_types:
        subset = results_df[results_df['tensor_type'] == tensor_type]
        Z_true = subset['Z_true'].iloc[0]

        print(f"\n[{tensor_type.upper()}]")
        print(f"  Z_true: {Z_true:.6e}")
        print(f"  rel_error range: {subset['rel_error'].min():.2e} to {subset['rel_error'].max():.2e}")
        print(f"  geom mean rel_error: {gmean_series(subset['rel_error'].dropna()):.2e}")
        print(f"  median rel_error: {subset['rel_error'].median():.2e}")

        # best configuration
        best_idx = subset.groupby(['A', 'B'])['rel_error'].apply(gmean_series).idxmin()
        best_A, best_B = best_idx
        best_C = args.N[0] // (best_A * best_B)
        best_err = subset.groupby(['A', 'B'])['rel_error'].apply(gmean_series).min()
        print(f"  best (A,B,C): ({best_A}, {best_B}, {best_C}) -> gmean_err = {best_err:.2e}")

    # Generate plots
    if not args.no_plots:
        for N_val in args.N:
            for tensor_type in args.tensor_types:
                # Geometric mean relative error
                save_path = f"{args.plots_dir}/mps_{tensor_type}_gmean_rel_error_N{N_val}.png" if args.plots_dir else None
                plot_cost_heatmap(results_df, N_value=N_val, tensor_type=tensor_type,
                                metric='rel_error', aggregation='gmean', save_path=save_path)

                # Variance of log relative error
                save_path = f"{args.plots_dir}/mps_{tensor_type}_var_log_rel_error_N{N_val}.png" if args.plots_dir else None
                plot_variance_heatmap(results_df, N_value=N_val, tensor_type=tensor_type,
                                    metric='rel_error', mode='logvar', save_path=save_path)

                # Geometric std dev
                save_path = f"{args.plots_dir}/mps_{tensor_type}_gstd_rel_error_N{N_val}.png" if args.plots_dir else None
                plot_variance_heatmap(results_df, N_value=N_val, tensor_type=tensor_type,
                                    metric='rel_error', mode='gstd', save_path=save_path)


if __name__ == "__main__":
    main()
