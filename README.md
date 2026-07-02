# AIS-MCMC Tensor Network Contraction

Annealed Importance Sampling (AIS) with MCMC to estimate the partition function
(trace) of a discrete tensor network. Intermediate distributions are tempered by
an inverse temperature $\beta \in [0, 1]$, annealing from an easy-to-sample base
distribution to the target defined by the network.

**Authors:** Aniket Deshpande, Sreevardhan Atyam, Qizhao Huang, Edgar Solomonik —
University of Illinois Urbana-Champaign

## Install

```bash
pip install -r requirements.txt
```

## Usage

```python
import networkx as nx
import numpy as np
from src.algorithm import TensorNetwork, run_multiple_chains

G = nx.Graph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
tensors = {
    'A': (np.random.rand(3, 3) + 1e-6, ['i', 'j']),
    'B': (np.random.rand(3, 3) + 1e-6, ['j', 'k']),
    'C': (np.random.rand(3, 3) + 1e-6, ['k', 'l']),
    'D': (np.random.rand(3, 3) + 1e-6, ['l', 'i']),
}
tn = TensorNetwork(G, tensors)

betas = np.linspace(0.0, 1.0, 200)
mean_Z, std_Z = run_multiple_chains(tn, betas, n_chains=100, iters=20000, verbose=True)
print(f"Z ≈ {mean_Z:.6e} ± {std_Z:.6e}")
```

## Layout

- `src/algorithm.py` — core AIS/MCMC implementation (`TensorNetwork`,
  `run_multiple_chains`, `contract_tensor_network`).
- `scripts/` — cost/heatmap experiments for 2×2 ring, 3×3 grid, and periodic MPS.
- `tests/` — network builders and validation against exact contraction.

## Tests

```bash
python tests/test_2x2_ring.py
python tests/test_3x3_grid.py
```
