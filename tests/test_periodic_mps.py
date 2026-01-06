# Test AIS on periodic MPS tensor network

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import networkx as nx
from src.algorithm import TensorNetwork, run_multiple_chains, estimate_contraction

# entry model constructors
def _make_core_gaussian(dim: int, rng: np.random.Generator, eps: float) -> np.ndarray:
    # old behavior: N(1, 0.1) but clipped to be nonnegative
    A = rng.normal(loc=1.0, scale=0.1, size=(dim, dim)).astype(np.float64)
    return np.clip(A, eps, None)

def _make_core_uniform1(dim: int, rng: np.random.Generator, jitter: float, eps: float) -> np.ndarray:
    # uniform around 1, nonnegative
    low, high = 1.0 - jitter, 1.0 + jitter
    A = rng.uniform(low, high, size=(dim, dim)).astype(np.float64)
    return np.clip(A, eps, None)

def _make_core_diagexp(dim: int, alpha: float, eps: float) -> np.ndarray:
    # exponentially-decaying diagonal
    # A[k, k] = exp(-alpha * k), off-diag = eps
    A = np.full((dim, dim), eps, dtype=np.float64)
    diag = np.exp(-alpha * np.arange(dim, dtype=np.float64))
    A[np.diag_indices(dim)] = diag
    return A

def _make_core_spikes(
    dim: int, 
    rng: np.random.Generator,
    spike_factor: float,
    jitter: float,
    eps: float,
) -> np.ndarray:
    # base uniform around 1 with multiplicative spikes on the full diagonal
    A = _make_core_uniform1(dim, rng, jitter, eps)
    A[np.diag_indices(dim)] *= spike_factor
    return A

def _normalize_tensor_type(tensor_type: str) -> str:
    # accepts
    # 'gaussian', 'normal', 'uniform1', 'diagexp', 'spikes'
    # also toleratoes 'mps_spikes', 'perioidc_mps_diagexp', etc.
    t = (tensor_type or "").strip().lower()
    # allow "foo_bar_baz" -> last token if it matches
    tokens = [tok for tok in t.replace("-", "_").split("_") if tok]
    candidates = tokens[::-1] if tokens else [t]

    for c in candidates:
        if c in {"gaussian", "normal"}:
            return "gaussian"
        if c in {"uniform1", "uniform"}:
            return "uniform1"
        if c in {"diagexp", "expdiag", "exp_diagonal", "exp"}:
            return "diagexp"
        if c in {"spikes", "spike"}:
            return "spikes"

    raise ValueError(
        f"Unknown tensor_type='{tensor_type}'. "
        "Supported: gaussian, uniform1, diagexp, spikes."
    )

# periodic MPS / tensor ring

def build_periodic_mps(
    n_sites: int = 16,
    dim: int = 3,
    tensor_type: str = "gaussian",
    seed: int = 42,
    eps: float = 1e-6,
    alpha: float = 0.6,
    spike_factor: float = 10.0,
    jitter: float = 0.1,
) -> tuple[nx.Graph, dict[str, tuple[np.ndarray, list[str]]]]:
    """
    Build a closed periodic MPS (tensor ring) as a cycle graph with rank-2 tensors at each node.

    Nodes:  A0, A1, ..., A{L-1}
    Bonds:  b0, b1, ..., b{L-1}  (bond bi connects Ai -- A(i+1))
    Tensor at site i has indices [b_{i-1}, b_i] (left, right), shape (dim, dim).

    Returns:
        G: networkx.Graph on the nodes (topology only; indices live in tensors)
        tensors: dict[name] = (ndarray, [index_labels])
    """
    if n_sites < 2:
        raise ValueError("n_sites must be >= 2 for a periodic MPS.")

    model = _normalize_tensor_type(tensor_type)
    rng = np.random.default_rng(seed)

    # topology
    G = nx.Graph()
    node_names = [f"A{i}" for i in range(n_sites)]
    for name in node_names:
        G.add_node(name)
    for i in range(n_sites):
        G.add_edge(node_names[i], node_names[(i + 1) % n_sites])

    # indices (bonds)
    def bond(i: int) -> str:
        return f"b{i % n_sites}"

    tensors: dict[str, tuple[np.ndarray, list[str]]] = {}

    for i, name in enumerate(node_names):
        left = bond(i - 1)
        right = bond(i)

        if model == "gaussian":
            core = _make_core_gaussian(dim=dim, rng=rng, eps=eps)
        elif model == "uniform1":
            core = _make_core_uniform1(dim=dim, rng=rng, jitter=jitter, eps=eps)
        elif model == "diagexp":
            core = _make_core_diagexp(dim=dim, alpha=alpha, eps=eps)
        elif model == "spikes":
            core = _make_core_spikes(dim=dim, rng=rng, spike_factor=spike_factor, jitter=jitter, eps=eps)
        else:
            # _normalize_tensor_type should prevent this
            raise RuntimeError(f"Unhandled model: {model}")

        tensors[name] = (core, [left, right])

    return G, tensors



def exact_contract_periodic_mps(
    tensors: dict[str, tuple[np.ndarray, list[str]]],
    n_sites: int,
    dim: int,
) -> float:
    """
    Exact contraction for this specific periodic MPS construction:
      Z = Tr(A0 A1 ... A_{L-1})
    Assumes node names are A0..A{L-1}.
    """
    prod = np.eye(dim, dtype=float)
    for i in range(n_sites):
        prod = prod @ tensors[f"A{i}"][0]
    return float(np.trace(prod))



def create_test_tensor_network(
    tensor_type: str = "gaussian",
    dim: int = 3,
    seed: int = 42,
    eps: float = 1e-6,
    alpha: float = 0.6,
    spike_factor: float = 10.0,
    jitter: float = 0.1,
    n_sites: int = 16,
):
    """
    Periodic MPS builder with configurable tensor entry model via `tensor_type`.

    tensor_type options:
      - "gaussian"   (old behavior: N(1, 0.1), clipped >= eps)
      - "uniform1"   (U[1-jitter, 1+jitter], clipped >= eps)
      - "diagexp"    (diagonal exp(-alpha*k), off-diagonal eps)
      - "spikes"     (uniform1 + diagonal * spike_factor)

    dim: bond dimension for every index
    n_sites: length of the ring
    """
    return build_periodic_mps(
        n_sites=n_sites,
        dim=dim,
        tensor_type=tensor_type,
        seed=seed,
        eps=eps,
        alpha=alpha,
        spike_factor=spike_factor,
        jitter=jitter,
    )


def make_logspace_betas(A: int) -> np.ndarray:
    """Generate logspace beta schedule for better early-beta resolution."""
    betas = 1.0 - np.logspace(0, np.log10(1e-6), A)
    betas[0] = 0.0
    betas[-1] = 1.0
    return np.sort(betas)


def test_periodic_mps(
    tensor_type: str = "gaussian",
    n_sites: int = 16,
    dim: int = 3,
    A: int = 200,
    B: int = 400,
    C: int = 200,
    seed: int = 42,
    eps: float = 1e-6,
    alpha: float = 0.6,
    spike_factor: float = 10.0,
    jitter: float = 0.1,
    show_diagnostics: bool = True,
):
    """
    Test AIS on periodic MPS (tensor ring) with configurable tensor entry model.
    
    Args:
        tensor_type: Entry model - "gaussian", "uniform1", "diagexp", or "spikes"
        n_sites: Number of sites in the periodic MPS (ring length)
        dim: Bond dimension for each index
        A: Number of beta values in annealing schedule
        B: Number of parallel chains
        C: Number of iterations per beta step
        seed: Random seed for reproducibility
        eps: Minimum value floor for tensor entries
        alpha: Decay rate for diagexp model
        spike_factor: Multiplicative factor for diagonal in spikes model
        jitter: Half-width of uniform distribution for uniform1/spikes models
        show_diagnostics: Whether to show detailed output
    
    Returns:
        mean_Z: Mean estimate of partition function
        std_Z: Standard deviation of estimate
        rel_error: Relative error compared to exact result
    """
    burns = max(0, min(C // 10, C - 1))
    
    if show_diagnostics:
        print(f"\n[info] building periodic MPS (n_sites={n_sites}, dim={dim}, type={tensor_type})")
    
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
    
    if show_diagnostics:
        print("[info] performing exact contraction...")
    TRUE_Z = exact_contract_periodic_mps(tensors, n_sites, dim)
    if show_diagnostics:
        print(f"[info] exact Z = {TRUE_Z:.12e}")

    tn = TensorNetwork(G, tensors)
    betas = make_logspace_betas(A)

    if show_diagnostics:
        print(f"\n[info] running AIS (A={A}, B={B}, C={C}, burns={burns})")
        print(f"[info] using logspace beta schedule")
    
    mean_Z, std_Z = run_multiple_chains(
        tn, betas,
        n_chains=B,
        iters=C,
        burns=burns,
        Z_true=TRUE_Z if show_diagnostics else None,
        verbose=show_diagnostics
    )

    rel_error = abs(mean_Z - TRUE_Z) / abs(TRUE_Z) if TRUE_Z != 0 else float('inf')
    
    if show_diagnostics:
        print("\n" + "=" * 60)
        print(f"[summary] Periodic MPS ({tensor_type}) Results")
        print("=" * 60)
        print(f"[config] n_sites={n_sites}, dim={dim}, type={tensor_type}")
        print(f"[result] exact Z      : {TRUE_Z:.12e}")
        print(f"[result] estimated Z  : {mean_Z:.12e} ± {std_Z:.12e}")
        print(f"[result] rel error    : {rel_error:.6e}")
        print("=" * 60)

    return mean_Z, std_Z, rel_error


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AIS on periodic MPS tensor network")
    parser.add_argument("--type", type=str, default="gaussian",
                        choices=["gaussian", "uniform1", "diagexp", "spikes"],
                        help="Tensor entry model type")
    parser.add_argument("--n_sites", type=int, default=16, help="Number of sites in the ring")
    parser.add_argument("--dim", type=int, default=3, help="Bond dimension")
    parser.add_argument("-A", type=int, default=200, help="Number of beta values")
    parser.add_argument("-B", type=int, default=400, help="Number of parallel chains")
    parser.add_argument("-C", type=int, default=200, help="Iterations per beta step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.6, help="Decay rate for diagexp")
    parser.add_argument("--spike_factor", type=float, default=10.0, help="Spike factor for spikes model")
    parser.add_argument("--jitter", type=float, default=0.1, help="Jitter for uniform1/spikes")
    
    args = parser.parse_args()
    
    test_periodic_mps(
        tensor_type=args.type,
        n_sites=args.n_sites,
        dim=args.dim,
        A=args.A,
        B=args.B,
        C=args.C,
        seed=args.seed,
        alpha=args.alpha,
        spike_factor=args.spike_factor,
        jitter=args.jitter,
    )