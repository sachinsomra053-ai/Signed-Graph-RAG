"""Signed community detection (factions).

Standard GraphRAG uses Leiden, which assumes non-negative weights. Here we
use a greedy correlation-clustering heuristic: put positive edges *inside*
communities and negative edges *between* them. The number of communities is
not fixed in advance.

Objective (to maximise):  sum of signed weights of edges inside communities.
"""

from __future__ import annotations

import random
from collections import defaultdict

import networkx as nx


def signed_communities(g: nx.Graph, max_iter: int = 50, seed: int = 42,
                       weight: str = "weight") -> dict[str, int]:
    rng = random.Random(seed)
    nodes = list(g.nodes())
    # start with every node in its own community
    comm = {n: i for i, n in enumerate(nodes)}

    for _ in range(max_iter):
        moved = False
        rng.shuffle(nodes)
        for n in nodes:
            # signed weight from n to each neighbouring community
            gain: dict[int, float] = defaultdict(float)
            for nbr, d in g[n].items():
                if nbr != n:
                    gain[comm[nbr]] += d.get(weight, d.get("sign", 0))
            current = comm[n]
            stay = gain.get(current, 0.0)
            best_c, best_g = current, stay
            for c, w in gain.items():
                if w > best_g + 1e-9:
                    best_c, best_g = c, w
            # if n only has negative ties to its community, isolate it
            if best_c == current and stay < 0:
                best_c = max(comm.values()) + 1
            if best_c != current:
                comm[n] = best_c
                moved = True
        if not moved:
            break

    # relabel 0..k-1, biggest community first
    sizes = defaultdict(int)
    for c in comm.values():
        sizes[c] += 1
    order = {c: i for i, c in enumerate(sorted(sizes, key=lambda c: -sizes[c]))}
    return {n: order[c] for n, c in comm.items()}


def community_stats(g: nx.Graph, comm: dict[str, int]) -> list[dict]:
    """Per-community internal cohesion and external hostility."""
    groups: dict[int, list[str]] = defaultdict(list)
    for n, c in comm.items():
        groups[c].append(n)
    stats = []
    for c, members in sorted(groups.items()):
        inside_pos = inside_neg = out_pos = out_neg = 0
        for u, v, d in g.edges(data=True):
            s = d.get("sign", 0)
            in_u, in_v = comm[u] == c, comm[v] == c
            if in_u and in_v:
                inside_pos += s > 0
                inside_neg += s < 0
            elif in_u or in_v:
                out_pos += s > 0
                out_neg += s < 0
        stats.append({
            "community": c, "members": sorted(members),
            "internal_positive": inside_pos, "internal_negative": inside_neg,
            "external_positive": out_pos, "external_negative": out_neg,
        })
    return stats


def summarize_community(g: nx.Graph, members: list[str], comm: dict[str, int],
                        max_evidence: int = 6) -> str:
    """Plain-text summary used as GraphRAG 'community report'. Feed it to an
    LLM for a prose summary, or use it directly as retrieval context."""
    mset = set(members)
    lines = [f"Faction members: {', '.join(sorted(members))}"]
    allies, rivals = [], []
    for u, v, d in g.edges(data=True):
        if u in mset and v in mset:
            (allies if d["sign"] > 0 else rivals).append((u, v, d))
    if allies:
        lines.append("Internal alliances: " + "; ".join(f"{u} + {v}" for u, v, _ in allies))
    if rivals:
        lines.append("Internal tensions: " + "; ".join(f"{u} vs {v}" for u, v, _ in rivals))
    hostile = defaultdict(int)
    for u, v, d in g.edges(data=True):
        if (u in mset) != (v in mset) and d["sign"] < 0:
            other = v if u in mset else u
            hostile[comm.get(other, -1)] += 1
    if hostile:
        lines.append("Main opponents: " + ", ".join(
            f"community {c} ({k} negative ties)" for c, k in sorted(hostile.items(), key=lambda x: -x[1])))
    ev = []
    for u, v, d in g.edges(data=True):
        if u in mset or v in mset:
            for r in d["relations"][:1]:
                if r.evidence:
                    ev.append(f"[{r.article_id}] {r.evidence}")
    if ev:
        lines.append("Evidence:\n  - " + "\n  - ".join(ev[:max_evidence]))
    return "\n".join(lines)
