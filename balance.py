"""Structural balance analysis for signed graphs.

Heider / Cartwright-Harary balance theory:
  a triangle is BALANCED when the product of its three edge signs is positive
      + + +   "friend of my friend is my friend"
      + - -   "enemy of my enemy is my friend"
  and UNBALANCED when the product is negative
      + + -   two of my friends are enemies  -> tension / contradiction
      - - -   everyone hates everyone        -> weak balance allows this

In a GraphRAG setting an unbalanced triangle is a signal that either
(a) the political situation is genuinely unstable / shifting, or
(b) sources disagree, or (c) the extractor got a sign wrong.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import numpy as np


@dataclass
class Triangle:
    nodes: tuple[str, str, str]
    signs: tuple[int, int, int]  # signs of (a-b, b-c, a-c)

    @property
    def balanced(self) -> bool:
        return self.signs[0] * self.signs[1] * self.signs[2] > 0

    @property
    def pattern(self) -> str:
        return "".join("+" if s > 0 else "-" for s in sorted(self.signs, reverse=True))

    def describe(self) -> str:
        a, b, c = self.nodes
        s1, s2, s3 = ("+" if s > 0 else "-" for s in self.signs)
        return f"{a} ({s1}) {b} ({s2}) {c} ({s3}) {a}"


@dataclass
class BalanceReport:
    triangles: list[Triangle]
    balance_ratio: float                  # share of balanced triangles
    pattern_counts: dict[str, int]
    camps: dict[str, int]                 # node -> 0/1, best two-faction split
    frustration: int                      # edges violating that split
    frustrated_edges: list[tuple[str, str, int]]
    suspicious_edges: list[tuple[str, str, int]] = field(default_factory=list)

    @property
    def unbalanced(self) -> list[Triangle]:
        return [t for t in self.triangles if not t.balanced]

    def to_dict(self) -> dict:
        return {
            "n_triangles": len(self.triangles),
            "balance_ratio": round(self.balance_ratio, 3),
            "pattern_counts": self.pattern_counts,
            "unbalanced_triangles": [t.describe() for t in self.unbalanced],
            "camps": self.camps,
            "frustration": self.frustration,
            "frustrated_edges": [list(e) for e in self.frustrated_edges],
            "suspicious_edges": [list(e) for e in self.suspicious_edges],
        }


# ---------------------------------------------------------------- triangles
def signed_triangles(g: nx.Graph) -> list[Triangle]:
    """Enumerate all triangles whose three edges are signed (+1/-1)."""
    out: list[Triangle] = []
    signed_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("sign", 0) != 0]
    h = nx.Graph()
    h.add_edges_from(signed_edges)
    order = {n: i for i, n in enumerate(sorted(h.nodes()))}
    for u in h.nodes():
        nbrs = [v for v in h[u] if order[v] > order[u]]
        for v, w in combinations(sorted(nbrs, key=order.get), 2):
            if h.has_edge(v, w):
                s = (g[u][v]["sign"], g[v][w]["sign"], g[u][w]["sign"])
                out.append(Triangle((u, v, w), s))
    return out


# ------------------------------------------------------ two-camp partition
def two_camp_partition(g: nx.Graph, n_restarts: int = 10, seed: int = 0) -> dict[str, int]:
    """Approximate the minimum-frustration split into two factions.

    Start from the leading eigenvector of the signed adjacency matrix, then
    greedily move single nodes while frustration decreases. Exact
    minimisation is NP-hard, this heuristic works well on news-sized graphs.
    """
    nodes = list(g.nodes())
    if not nodes:
        return {}
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for u, v, d in g.edges(data=True):
        A[idx[u], idx[v]] = A[idx[v], idx[u]] = d.get("sign", 0)

    rng = np.random.default_rng(seed)
    starts = []
    if len(nodes) > 1:
        _, vecs = np.linalg.eigh(A)
        starts.append((vecs[:, -1] >= 0).astype(int))
    for _ in range(n_restarts):
        starts.append(rng.integers(0, 2, len(nodes)))

    best, best_f = None, float("inf")
    for x in starts:
        x = _local_search(A, x.copy())
        f = _frustration(A, x)
        if f < best_f:
            best, best_f = x, f
    return {n: int(best[idx[n]]) for n in nodes}


def _frustration(A: np.ndarray, x: np.ndarray) -> int:
    same = (x[:, None] == x[None, :])
    bad = ((A > 0) & ~same) | ((A < 0) & same)
    return int(np.triu(bad, 1).sum())


def _local_search(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    improved = True
    while improved:
        improved = False
        for i in range(len(x)):
            s = np.where(x == x[i], 1, -1)  # +1 same camp, -1 other camp
            # gain of flipping i = current violations - violations after flip
            row = A[i] * s
            row[i] = 0
            if row.sum() < 0:  # more violated than satisfied edges -> flip
                x[i] = 1 - x[i]
                improved = True
    return x


# ------------------------------------------------------------------ report
def analyze_balance(g: nx.Graph) -> BalanceReport:
    tris = signed_triangles(g)
    n_bal = sum(t.balanced for t in tris)
    ratio = n_bal / len(tris) if tris else 1.0
    patterns = Counter(t.pattern for t in tris)

    signed_g = g.edge_subgraph(
        [(u, v) for u, v, d in g.edges(data=True) if d.get("sign", 0) != 0]
    ).copy()
    camps = two_camp_partition(signed_g)
    frustrated = []
    for u, v, d in signed_g.edges(data=True):
        same = camps[u] == camps[v]
        if (d["sign"] > 0 and not same) or (d["sign"] < 0 and same):
            frustrated.append((u, v, d["sign"]))

    # Edges that appear in many unbalanced triangles are the best candidates
    # for extraction errors or genuinely shifting alliances.
    edge_hits: Counter = Counter()
    for t in tris:
        if not t.balanced:
            a, b, c = t.nodes
            for e, s in (((a, b), t.signs[0]), ((b, c), t.signs[1]), ((a, c), t.signs[2])):
                edge_hits[(*sorted(e), s)] += 1
    suspicious = [e for e, _ in edge_hits.most_common(10)]

    return BalanceReport(
        triangles=tris, balance_ratio=ratio, pattern_counts=dict(patterns),
        camps=camps, frustration=len(frustrated), frustrated_edges=frustrated,
        suspicious_edges=suspicious,
    )
