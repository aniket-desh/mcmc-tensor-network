# test_tree_ttn.py
# Test AIS on a TREE Tensor Network (binary TTN / TTNS-style topology)
#
# You asked for:
#   - control depth `d`
#   - leaves are vectors (rank-1), each shares exactly ONE index with its parent
#   - whole network is closed -> scalar contraction
#
# Topology:
#   Full binary tree of depth `depth` (root at level 0, leaves at level `depth`)
#   - Internal non-root nodes: rank-3 tensors with legs [parent, left_child, right_child]
#   - Root: rank-2 tensor with legs [left_child, right_child]
#   - Leaves: rank-1 vectors with leg [parent]
#
# Entry models via `tensor_type`: gaussian, uniform1, diagexp, spikes

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import networkx as nx

from src.algorithm import TensorNetwork, run_multiple_chains


# -------------------------
# entry model constructors
# -------------------------

def _normalize_tensor_type(tensor_type: str) -> str:
    t = (tensor_type or "").strip().lower()
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
        f"Unknown tensor_type='{tensor_type}'. Supported: gaussian, uniform1, diagexp, spikes."
    )


def _make_tensor_gaussian(shape: tuple[int, ...], rng: np.random.Generator, eps: float) -> np.ndarray:
    X = rng.normal(loc=1.0, scale=0.1, size=shape).astype(np.float64)
    return np.clip(X, eps, None)


def _make_tensor_uniform1(shape: tuple[int, ...], rng: np.random.Generator, jitter: float, eps: float) -> np.ndarray:
    lo, hi = 1.0 - jitter, 1.0 + jitter
    X = rng.uniform(lo, hi, size=shape).astype(np.float64)
    return np.clip(X, eps, None)


def _make_tensor_diagexp(dim: int, rank: int, alpha: float, eps: float) -> np.ndarray:
    """
    "Full diagonal" for any rank r:
      T[k,k,...,k] = exp(-alpha*k), all other entries = eps
    For rank=1 this is just a vector v[k]=exp(-alpha*k).
    """
    shape = (dim,) * rank
    T = np.full(shape, eps, dtype=np.float64)
    diag = np.exp(-alpha * np.arange(dim, dtype=np.float64))

    if rank == 1:
        T[:] = diag
        return T

    for k in range(dim):
        T[(k,) * rank] = diag[k]
    return T


def _make_tensor_spikes(dim: int, rank: int, rng: np.random.Generator, spike_factor: float, jitter: float, eps: float) -> np.ndarray:
    """
    Base uniform around 1, then multiply the "full diagonal" entries by spike_factor.
    For rank=1 (vector), we spike a single coordinate k=0 (deterministic) to keep behavior stable.
    """
    T = _make_tensor_uniform1((dim,) * rank, rng=rng, jitter=jitter, eps=eps)

    if rank == 1:
        T[0] *= spike_factor
        return T

    for k in range(dim):
        T[(k,) * rank] *= spike_factor
    return T


def _make_tensor_by_type(
    *,
    dim: int,
    rank: int,
    model: str,
    rng: np.random.Generator,
    eps: float,
    alpha: float,
    spike_factor: float,
    jitter: float,
) -> np.ndarray:
    if model == "gaussian":
        return _make_tensor_gaussian((dim,) * rank, rng=rng, eps=eps)
    if model == "uniform1":
        return _make_tensor_uniform1((dim,) * rank, rng=rng, jitter=jitter, eps=eps)
    if model == "diagexp":
        return _make_tensor_diagexp(dim=dim, rank=rank, alpha=alpha, eps=eps)
    if model == "spikes":
        return _make_tensor_spikes(dim=dim, rank=rank, rng=rng, spike_factor=spike_factor, jitter=jitter, eps=eps)
    raise RuntimeError(f"Unhandled model: {model}")


# -------------------------
# binary TTN builder
# -------------------------

def build_binary_ttn(
    depth: int = 4,
    dim: int = 3,
    tensor_type: str = "gaussian",
    seed: int = 42,
    eps: float = 1e-6,
    alpha: float = 0.6,
    spike_factor: float = 10.0,
    jitter: float = 0.1,
) -> tuple[nx.Graph, dict[str, tuple[np.ndarray, list[str]]]]:
    """
    Build a closed binary-tree TN with leaf vectors.

    depth = number of edges from root to leaf.
      leaves = 2**depth
      internal nodes (including root) = 2**depth - 1

    Node naming:
      root:          I0_0
      internal:      I{level}_{pos} for level=0..depth-1
      leaves:        L{pos} for pos=0..2**depth - 1

    Edge labels:
      b_{parent}_{child}  (unique per edge)

    Tensor index order:
      - root I0_0:                 [b_{I0_0}_{I1_0}, b_{I0_0}_{I1_1}]  (rank 2)
      - internal non-root Iℓ_p:    [b_{parent}_{Iℓ_p}, b_{Iℓ_p}_{left}, b_{Iℓ_p}_{right}] (rank 3)
      - leaf Lp:                   [b_{parent}_{Lp}] (rank 1)
    """
    if depth < 1:
        raise ValueError("depth must be >= 1 (so there is at least one leaf layer).")
    if dim < 2:
        raise ValueError("dim must be >= 2.")

    model = _normalize_tensor_type(tensor_type)
    rng = np.random.default_rng(seed)

    G = nx.Graph()
    tensors: dict[str, tuple[np.ndarray, list[str]]] = {}

    def iname(level: int, pos: int) -> str:
        return f"I{level}_{pos}"

    def lname(pos: int) -> str:
        return f"L{pos}"

    def bond(parent: str, child: str) -> str:
        return f"b_{parent}_{child}"

    # add all internal nodes
    for level in range(depth):
        for pos in range(2**level):
            G.add_node(iname(level, pos))

    # add leaves
    for pos in range(2**depth):
        G.add_node(lname(pos))

    # add edges + build tensors
    # build internal connectivity
    for level in range(depth):
        for pos in range(2**level):
            this = iname(level, pos)

            # children
            if level == depth - 1:
                left = lname(2 * pos)
                right = lname(2 * pos + 1)
            else:
                left = iname(level + 1, 2 * pos)
                right = iname(level + 1, 2 * pos + 1)

            G.add_edge(this, left)
            G.add_edge(this, right)

    # now define tensors in a second pass so every node exists
    # root tensor
    root = iname(0, 0)
    if depth == 1:
        c1, c2 = lname(0), lname(1)
    else:
        c1, c2 = iname(1, 0), iname(1, 1)
    root_labels = [bond(root, c1), bond(root, c2)]
    root_tensor = _make_tensor_by_type(
        dim=dim, rank=2, model=model, rng=rng, eps=eps, alpha=alpha, spike_factor=spike_factor, jitter=jitter
    )
    tensors[root] = (root_tensor, root_labels)

    # internal non-root
    for level in range(1, depth):
        for pos in range(2**level):
            this = iname(level, pos)
            parent = iname(level - 1, pos // 2)

            # children
            if level == depth - 1:
                left = lname(2 * pos)
                right = lname(2 * pos + 1)
            else:
                left = iname(level + 1, 2 * pos)
                right = iname(level + 1, 2 * pos + 1)

            labels = [bond(parent, this), bond(this, left), bond(this, right)]
            T = _make_tensor_by_type(
                dim=dim, rank=3, model=model, rng=rng, eps=eps, alpha=alpha, spike_factor=spike_factor, jitter=jitter
            )
            tensors[this] = (T, labels)

    # leaves (vectors)
    for pos in range(2**depth):
        leaf = lname(pos)
        parent = iname(depth - 1, pos // 2)
        labels = [bond(parent, leaf)]
        v = _make_tensor_by_type(
            dim=dim, rank=1, model=model, rng=rng, eps=eps, alpha=alpha, spike_factor=spike_factor, jitter=jitter
        )
        tensors[leaf] = (v, labels)

    return G, tensors


# -------------------------
# exact contraction on a tree (message passing)
# -------------------------

def exact_contract_ttn_tree(
    G: nx.Graph,
    tensors: dict[str, tuple[np.ndarray, list[str]]],
    dim: int,
    root: str = "I0_0",
) -> float:
    """
    Exact contraction on a tree in O(#nodes * dim^3) for binary TTN:
      - compute "messages" upward from leaves to root
      - leaf message is its vector
      - internal message is contraction of its tensor with child messages, leaving parent index uncontracted
      - root produces scalar
    """
    # Root the tree
    parent = {root: None}
    order = [root]
    for u in order:
        for v in G.neighbors(u):
            if v not in parent:
                parent[v] = u
                order.append(v)

    # helper: shared edge label between neighboring nodes
    def shared_label(u: str, v: str) -> str:
        labs_u = set(tensors[u][1])
        labs_v = set(tensors[v][1])
        inter = labs_u & labs_v
        if len(inter) != 1:
            raise RuntimeError(f"Expected exactly 1 shared label between {u} and {v}, got {inter}")
        return next(iter(inter))

    # children lists
    children = {u: [] for u in order}
    for u in order:
        pu = parent[u]
        for v in G.neighbors(u):
            if v == pu:
                continue
            children[u].append(v)

    # postorder (children before parent)
    post = order[::-1]

    msg_to_parent: dict[str, np.ndarray] = {}

    for u in post:
        pu = parent[u]
        T, labs = tensors[u]

        if len(children[u]) == 0:
            # leaf: rank-1 vector over its parent bond
            msg_to_parent[u] = T.astype(np.float64, copy=False)
            continue

        # internal or root
        ch = children[u]
        if pu is None:
            # root: should have 2 children, rank-2 tensor
            c1, c2 = ch[0], ch[1]
            lab1 = shared_label(u, c1)
            lab2 = shared_label(u, c2)

            a1 = labs.index(lab1)
            a2 = labs.index(lab2)
            Tperm = np.transpose(T, (a1, a2))  # axes align with messages

            m1 = msg_to_parent[c1]
            m2 = msg_to_parent[c2]
            Z = np.einsum("ab,a,b->", Tperm, m1, m2)
            return float(Z)

        # non-root internal: rank-3 message to parent
        parent_lab = shared_label(u, pu)
        c1, c2 = ch[0], ch[1]
        lab1 = shared_label(u, c1)
        lab2 = shared_label(u, c2)

        ap = labs.index(parent_lab)
        a1 = labs.index(lab1)
        a2 = labs.index(lab2)

        Tperm = np.transpose(T, (ap, a1, a2))
        m1 = msg_to_parent[c1]
        m2 = msg_to_parent[c2]

        msg = np.einsum("pab,a,b->p", Tperm, m1, m2)
        msg_to_parent[u] = msg

    raise RuntimeError("Root was not processed correctly.")


# -------------------------
# AIS harness (mirrors your other tests)
# -------------------------

def make_logspace_betas(A: int) -> np.ndarray:
    betas = 1.0 - np.logspace(0, np.log10(1e-6), A)
    betas[0] = 0.0
    betas[-1] = 1.0
    return np.sort(betas)


def test_tree_ttn(
    tensor_type: str = "gaussian",
    depth: int = 4,
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
    burns = max(0, min(C // 10, C - 1))

    if show_diagnostics:
        n_leaves = 2**depth
        print(f"\n[info] building binary TTN (depth={depth}, leaves={n_leaves}, dim={dim}, type={tensor_type})")

    G, tensors = build_binary_ttn(
        depth=depth,
        dim=dim,
        tensor_type=tensor_type,
        seed=seed,
        eps=eps,
        alpha=alpha,
        spike_factor=spike_factor,
        jitter=jitter,
    )

    if show_diagnostics:
        print("[info] performing exact contraction (tree message passing)...")
    TRUE_Z = exact_contract_ttn_tree(G, tensors, dim=dim, root="I0_0")
    if show_diagnostics:
        print(f"[info] exact Z = {TRUE_Z:.12e}")

    tn = TensorNetwork(G, tensors)
    betas = make_logspace_betas(A)
    # betas = np.linspace(0, 1, A)

    if show_diagnostics:
        print(f"\n[info] running AIS (A={A}, B={B}, C={C}, burns={burns})")

    mean_Z, std_Z = run_multiple_chains(
        tn, betas,
        n_chains=B,
        iters=C,
        burns=burns,
        Z_true=TRUE_Z if show_diagnostics else None,
        verbose=show_diagnostics,
    )

    rel_error = abs(mean_Z - TRUE_Z) / abs(TRUE_Z) if TRUE_Z != 0 else float("inf")

    if show_diagnostics:
        print("\n" + "=" * 60)
        print(f"[summary] Tree TTN ({tensor_type}) Results")
        print("=" * 60)
        print(f"[config] depth={depth}, dim={dim}, type={tensor_type}")
        print(f"[result] exact Z      : {TRUE_Z:.12e}")
        print(f"[result] estimated Z  : {mean_Z:.12e} ± {std_Z:.12e}")
        print(f"[result] rel error    : {rel_error:.6e}")
        print("=" * 60)

    return mean_Z, std_Z, rel_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AIS on a binary Tree Tensor Network (TTN)")
    parser.add_argument("--type", type=str, default="gaussian",
                        choices=["gaussian", "uniform1", "diagexp", "spikes"],
                        help="Tensor entry model type")
    parser.add_argument("--depth", type=int, default=4, help="Tree depth (edges root->leaf)")
    parser.add_argument("--dim", type=int, default=3, help="Bond dimension")
    parser.add_argument("-A", type=int, default=200, help="Number of beta values")
    parser.add_argument("-B", type=int, default=400, help="Number of parallel chains")
    parser.add_argument("-C", type=int, default=200, help="Iterations per beta step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eps", type=float, default=1e-6, help="Floor for entries")
    parser.add_argument("--alpha", type=float, default=0.6, help="Decay rate for diagexp")
    parser.add_argument("--spike_factor", type=float, default=10.0, help="Spike factor")
    parser.add_argument("--jitter", type=float, default=0.1, help="Uniform jitter half-width")

    args = parser.parse_args()

    test_tree_ttn(
        tensor_type=args.type,
        depth=args.depth,
        dim=args.dim,
        A=args.A,
        B=args.B,
        C=args.C,
        seed=args.seed,
        eps=args.eps,
        alpha=args.alpha,
        spike_factor=args.spike_factor,
        jitter=args.jitter,
    )
