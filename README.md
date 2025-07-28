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

The repository includes several test cases that demonstrate different tensor network structures:

- `test_2x2_ring.py`: Computes Tr(ABCD) for a 2×2 ring of random positive matrices
- `test_3x3_grid.py`: Estimates the contraction of a 3×3 grid tensor network with known exact value
- `test_3x3_dd.py`: Estimates the contraction of a 3×3 grid with diagonally dominant tensors

Each test can be run with configurable parameters:

```python
# Example: 2x2 ring test
mean_Z, std_Z, rel_error = test_trace_ABCD(
    dim=3,                          # Tensor dimension
    betas=np.linspace(0, 1, 200),   # Beta schedule
    n_chains=5,                     # Number of parallel chains
    iters=20000,                    # MCMC iterations per beta step
    burns=1900,                     # Burn-in iterations
    show_diagnostics=True           # Show plots and detailed output
)
```

## Parameters

| Parameter           | Description                                    |
|---------------------|------------------------------------------------|
| `iters`             | MCMC iterations per beta step                 |
| `burns`             | Burn-in iterations before sampling            |
| `n_chains`          | Number of independent AIS chains              |
| `n_betas`           | Number of beta values in annealing schedule   |
| `betas`             | Temperature schedule from 0 to 1              |
| `show_diagnostics`  | Whether to show diagnostic plots and output   |

## Output

Each run produces:

- An estimate of the partition function Z and standard deviation
- Relative variance across chains
- Diagnostic plots for log-Z trajectories and weight dispersion
- Log-weight variance analysis across beta steps
- Per-chain relative error analysis (when exact value is known)