# AIS-MCMC Tensor Network Contraction

This repository implements an Annealed Importance Sampling (AIS) algorithm using Markov Chain Monte Carlo (MCMC) techniques to estimate the partition function (or trace) of a discrete tensor network.

The method applies a sequence of intermediate distributions parameterized by inverse temperatures $\beta \in [0, 1]$, transitioning from an easy-to-sample base distribution to the target distribution defined by the tensor network. At each β step, multiple MCMC rounds are performed to estimate local partition function ratios, which are then aggregated into the final estimate.

## Key Features

- Arbitrary discrete tensor networks specified as graphs (via NetworkX)
- AIS estimator with configurable $\beta$-ladder and multiple sampling rounds
- Glauber dynamics for local MCMC updates
- Weight variance diagnostics and convergence plots
- Exact contraction via `numpy.einsum` for validation on small examples
- Supports multiple independent AIS chains for variance estimation

## Example Usage

The repository includes two test cases:

- `test_2x2_ring.py`: Computes Tr(ABCD) for a 2×2 ring of random positive matrices.
- `test_3x3_grid.py`: Estimates the contraction of a 3×3 grid tensor network with known exact value.

## Parameters

| Parameter   | Description                            |
|-------------|----------------------------------------|
| `iters`     | MCMC iterations per AIS round          |
| `burns`     | Burn-in iterations before sampling     |
| `n_rounds`  | Number of samples per $\beta$-step           |
| `n_chains`  | Number of independent AIS chains       |
| `betas`     | Temperature schedule from 0 to 1       |

## Output

Each run produces:

- An estimate of the partition function Z and standard deviation
- Relative variance across chains
- Diagnostic plots for log-Z trajectories and weight dispersion