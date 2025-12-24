# AIS-MCMC Tensor Network Contraction

This repository implements an Annealed Importance Sampling (AIS) algorithm using Markov Chain Monte Carlo (MCMC) techniques to estimate the partition function (or trace) of a discrete tensor network.

The method applies a sequence of intermediate distributions parameterized by inverse temperatures $\beta \in [0, 1]$, transitioning from an easy-to-sample base distribution to the target distribution defined by the tensor network. At each β step, multiple MCMC rounds are performed to estimate local partition function ratios, which are then aggregated into the final estimate.

## Authors

Aniket Deshpande, Sreevardhan Atyam, Qizhao Huang, Edgar Solomonik  
University of Illinois Urbana-Champaign

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
- `numpy >= 1.21.0`
- `networkx >= 2.6.0`
- `matplotlib >= 3.4.0`
- `seaborn >= 0.11.0`

## Core Algorithm

The main algorithm is implemented in `src/algorithm.py`:

### TensorNetwork Class

Represents a discrete tensor network as a graph structure:
- `graph`: NetworkX graph representing the tensor network topology
- `tensors`: Dictionary mapping node names to `(tensor_array, index_labels)` tuples
- Automatically validates index dimensions and builds index-to-tensor mappings

### Key Functions

- **`evaluate_config(network, configs)`**: Evaluates tensor network configurations, returning the product of all tensor entries for given index assignments.

- **`update_edge(network, configs, idx, beta)`**: Performs a single Glauber dynamics update on a specified index. Computes conditional probabilities from all tensors touching the index and resamples according to the β-annealed distribution.

- **`estimate_contraction(net, betas, iters, burns, n_chains, verbose)`**: Main AIS algorithm that:
  - Initializes random configurations for all chains
  - For each β step, performs MCMC mixing under the previous β distribution
  - Computes incremental weights and accumulates log-partition function estimates
  - Returns partition function estimates, log-Z trajectories, and weight arrays

- **`run_multiple_chains(tn, betas, ...)`**: Convenience wrapper that runs `estimate_contraction` and returns mean and standard deviation of partition function estimates.

- **`contract_tensor_network(graph, tensors)`**: Exact contraction via `numpy.einsum` for validation on small examples.

## Test Suite

The repository includes comprehensive test cases demonstrating different tensor network structures:

### 2×2 Ring Network
- **`test_2x2_ring.py`**: Computes Tr(ABCD) for a 2×2 ring of random positive matrices
  - Tests the simplest non-trivial tensor network structure
  - Validates against exact trace computation

### 3×3 Grid Networks
- **`test_3x3_grid.py`**: 3×3 grid with Gaussian-distributed tensor entries
- **`test_3x3_dd.py`**: 3×3 grid with diagonally dominant tensors (configurable noise level)
- **`test_3x3_exp_diag.py`**: 3×3 grid with exponentially-decaying diagonal entries
- **`test_3x3_spikes.py`**: 3×3 grid with multiplicative spikes on hyper-diagonal entries
- **`test_3x3_uniform1.py`**: 3×3 grid with uniform entries around 1

Each test:
- Builds a tensor network with specified structure
- Computes exact contraction for validation
- Runs AIS with configurable parameters
- Reports mean estimate, standard deviation, and relative error

## Usage Example

```python
from src.algorithm import TensorNetwork, run_multiple_chains
import networkx as nx
import numpy as np

# Build a simple tensor network
G = nx.Graph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])

tensors = {
    'A': (np.random.rand(3, 3) + 1e-6, ['i', 'j']),
    'B': (np.random.rand(3, 3) + 1e-6, ['j', 'k']),
    'C': (np.random.rand(3, 3) + 1e-6, ['k', 'l']),
    'D': (np.random.rand(3, 3) + 1e-6, ['l', 'i'])
}

tn = TensorNetwork(G, tensors)

# Generate logspace beta schedule (better resolution at low β)
def make_logspace_betas(A):
    betas = 1.0 - np.logspace(0, np.log10(1e-6), A)
    betas[0] = 0.0
    betas[-1] = 1.0
    return np.sort(betas)

betas = make_logspace_betas(200)

# Run AIS
mean_Z, std_Z = run_multiple_chains(
    tn, betas,
    n_chains=100,      # Number of parallel chains
    iters=20000,       # MCMC iterations per beta step
    burns=1900,        # Burn-in iterations
    verbose=True
)

print(f"Estimated Z: {mean_Z:.6e} ± {std_Z:.6e}")
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `betas` | Array of inverse temperature values from 0 to 1 (annealing schedule) |
| `iters` | Number of MCMC iterations per beta step |
| `burns` | Burn-in iterations before sampling (currently unused in main algorithm) |
| `n_chains` | Number of independent AIS chains for variance estimation |
| `verbose` | Whether to print progress and diagnostic information |

## Algorithm Details

The AIS algorithm works as follows:

1. **Initialization**: Random configurations are sampled uniformly for all chains.

2. **Annealing Loop**: For each β step from β₀=0 to βₖ=1:
   - **Mixing**: All chains are mixed under the previous β distribution using Glauber dynamics (random edge updates).
   - **Weight Computation**: Incremental weights are computed as $w = \psi(\mathbf{x})^{\Delta\beta}$ where $\psi$ is the tensor network evaluation and $\Delta\beta$ is the β step size.
   - **Accumulation**: Log-partition function estimates are updated: $\log Z \leftarrow \log Z + \log w$.

3. **Final Estimate**: The partition function is estimated as $Z = \exp(\log Z_{\text{sum}} + \log Z_0)$ where $Z_0$ is the uniform base distribution normalization.

## Output

The algorithm returns:
- **Partition function estimates**: Array of Z estimates (one per chain)
- **Log-Z trajectories**: Evolution of log-partition function estimates across β steps
- **Weight arrays**: Incremental weights at each β step for diagnostic analysis

The `run_multiple_chains` wrapper provides:
- Mean and standard deviation of partition function estimates
- Relative error (when exact value is provided)

## Running Tests

Run individual test files:

```bash
python tests/test_2x2_ring.py
python tests/test_3x3_grid.py
python tests/test_3x3_dd.py
python tests/test_3x3_exp_diag.py
python tests/test_3x3_spikes.py
python tests/test_3x3_uniform1.py
```

Each test can be customized by modifying parameters in the test function calls.
