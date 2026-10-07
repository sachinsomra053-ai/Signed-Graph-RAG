"""Signed local search: find entities in the question, walk signed paths
between them, and build an evidence-rich context for the LLM."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import networkx as nx

from .balance import analyze_balance
from .schema import Entity


@dataclass
class SignedPath:
    nodes: list[str]
    signs: list[int]
    confidence: float

    @property
    def sign(self) -> int:
        return math.prod(self.signs) if self.signs else 0

    def describe(self) -> str:
        parts = [self.nodes[0]]
        for n, s in zip(self.nodes[1:], self.signs):
            parts.append(f" --({'+' if s > 0 else '-'})--> {n}")
        verdict = "implied ALLY (+)" if self.sign > 0 else "implied OPPONENT (-)"
        return f"{''.join(parts)}   => {verdict}, confidence {self.confidence:.2f}"


class SignedRetriever:
    def __init__(self, graph: nx.Graph, entities: list[Entity] | None = None):
        self.g = graph
        self.alias = {n.lower(): n for n in graph.nodes()}
        for e in entities or []:
            for a in (e.name, *e.aliases):
                if e.name in graph:
                    self.alias[a.lower()] = e.name

    # --------------------------------------------------------------- entities
    def find_entities(self, query: str) -> list[str]:
        q = query.lower()
        found: list[str] = []
        for alias in sorted(self.alias, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", q):
                name = self.alias[alias]
                if name not in found:
                    found.append(name)
                q = q.replace(alias, " ")
        return found

    # ------------------------------------------------------------------ paths
    def signed_paths(self, a: str, b: str, max_hops: int = 3, limit: int = 5) -> list[SignedPath]:
        if a not in self.g or b not in self.g:
            return []
        paths = []
        for p in nx.all_simple_paths(self.g, a, b, cutoff=max_hops):
            edges = [self.g[u][v] for u, v in zip(p, p[1:])]
            if any(e["sign"] == 0 for e in edges):
                continue
            signs = [e["sign"] for e in edges]
            # path confidence: product of per-edge certainty, decays with length
            conf = 1.0
            for e in edges:
                conf *= abs(e["weight"]) / max(e["pos"] + e["neg"], 1.0)
            conf *= 0.85 ** (len(edges) - 1)
            paths.append(SignedPath(p, signs, conf))
        paths.sort(key=lambda p: (len(p.nodes), -p.confidence))
        return paths[:limit]

    def neighborhood(self, n: str) -> tuple[list[str], list[str]]:
        allies = sorted(v for v in self.g[n] if self.g[n][v]["sign"] > 0)
        rivals = sorted(v for v in self.g[n] if self.g[n][v]["sign"] < 0)
        return allies, rivals

    # ---------------------------------------------------------------- context
    def build_context(self, query: str, max_hops: int = 3) -> str:
        ents = self.find_entities(query)
        if not ents:
            return "No known entities found in the question."
        out = [f"Entities in question: {', '.join(ents)}"]

        for e in ents:
            allies, rivals = self.neighborhood(e)
            out.append(f"\n## {e}\n  Allies/supporters: {', '.join(allies) or '-'}"
                       f"\n  Opponents/critics: {', '.join(rivals) or '-'}")
            for v in self.g[e]:
                d = self.g[e][v]
                if d["contested"]:
                    out.append(f"  CONTESTED relationship with {v}: "
                               f"{d['pos']:.1f} positive vs {d['neg']:.1f} negative evidence")

        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                out.append(f"\n## Connection {a} <-> {b}")
                if self.g.has_edge(a, b):
                    d = self.g[a][b]
                    out.append(f"  Direct edge sign: {'+' if d['sign'] > 0 else '-'} "
                               f"(net weight {d['weight']:+.2f})")
                    for r in d["relations"][:4]:
                        out.append(f"    [{r.article_id} {r.date}] {r.source} {r.relation} "
                                   f"{r.target} ({'+' if r.sign > 0 else '-' if r.sign < 0 else '0'}): "
                                   f"\"{r.evidence}\"")
                paths = self.signed_paths(a, b, max_hops=max_hops)
                for p in paths:
                    out.append("  Path: " + p.describe())
                if paths:
                    signs = {p.sign for p in paths}
                    if len(signs) > 1:
                        out.append("  WARNING: paths disagree in sign - the relationship is "
                                   "ambiguous or shifting.")

        # local balance check around the question's entities
        sub = self.g.subgraph(set(ents) | {n for e in ents for n in self.g[e]})
        rep = analyze_balance(sub)
        bad = [t for t in rep.unbalanced if set(t.nodes) & set(ents)]
        if bad:
            out.append("\n## Unbalanced triangles (possible contradictions or shifting alliances)")
            out.extend(f"  {t.describe()}" for t in bad[:5])
        return "\n".join(out)

    # ----------------------------------------------------------------- answer
    def answer(self, query: str, model: str | None = None) -> str:
        """Generate an answer with an LLM using the signed context."""
        context = self.build_context(query)
        prompt = (
            "Answer the question using ONLY the signed knowledge-graph context below. "
            "'+' edges mean cooperation/support, '-' edges mean conflict/opposition. "
            "Cite article ids. Point out contested relationships and contradictions "
            "instead of hiding them.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {query}"
        )
        import anthropic  # optional dependency

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model or os.getenv("SIGNED_GRAPHRAG_MODEL", "claude-sonnet-5-5"),
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
