"""Signed knowledge graph built from extracted relations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx

from .schema import SignedRelation


class SignedKnowledgeGraph:
    """Keeps every raw relation (with evidence) and exposes an aggregated,
    undirected signed graph for balance analysis and clustering.

    Aggregation: for each entity pair the signed weights (sign * confidence)
    of all relations are summed. The edge sign is the sign of that sum. Pairs
    whose positive and negative evidence roughly cancel are marked
    `contested` - often the most interesting edges in politics.
    """

    def __init__(self, contested_threshold: float = 0.35):
        self.relations: list[SignedRelation] = []
        self.entity_types: dict[str, str] = {}
        self.contested_threshold = contested_threshold

    # ------------------------------------------------------------------ build
    def add(self, rel: SignedRelation) -> None:
        self.relations.append(rel)

    def add_many(self, rels) -> None:
        for r in rels:
            self.add(r)

    def set_entity_types(self, mapping: dict[str, str]) -> None:
        self.entity_types.update(mapping)

    # -------------------------------------------------------------- views
    def directed(self) -> nx.MultiDiGraph:
        """Every relation as its own directed edge (for provenance)."""
        g = nx.MultiDiGraph()
        for r in self.relations:
            for n in (r.source, r.target):
                g.add_node(n, type=self.entity_types.get(n, "OTHER"))
            g.add_edge(r.source, r.target, **r.to_dict())
        return g

    def signed(self, include_neutral: bool = False) -> nx.Graph:
        """Aggregated undirected signed graph.

        Edge attributes: sign (+1/-1/0), weight (net signed weight),
        pos, neg (evidence mass on each side), contested (bool),
        relations (list of SignedRelation).
        """
        buckets: dict[tuple[str, str], list[SignedRelation]] = defaultdict(list)
        for r in self.relations:
            if r.source == r.target:
                continue
            key = tuple(sorted((r.source, r.target)))
            buckets[key].append(r)

        g = nx.Graph()
        for (a, b), rels in buckets.items():
            pos = sum(r.confidence for r in rels if r.sign > 0)
            neg = sum(r.confidence for r in rels if r.sign < 0)
            net = pos - neg
            if pos == 0 and neg == 0:
                if not include_neutral:
                    continue
                sign = 0
            else:
                sign = 1 if net > 0 else -1 if net < 0 else 0
                if sign == 0:
                    # Perfect tie: the most recent signed relation wins, because
                    # in politics the latest move usually reflects the current state.
                    latest = max((r for r in rels if r.sign != 0), key=lambda r: r.date)
                    sign = latest.sign
            contested = pos > 0 and neg > 0 and abs(net) / (pos + neg) < self.contested_threshold
            for n in (a, b):
                g.add_node(n, type=self.entity_types.get(n, "OTHER"))
            g.add_edge(a, b, sign=sign, weight=net, pos=pos, neg=neg,
                       contested=contested, relations=rels)
        return g

    # -------------------------------------------------------------- stats
    def summary(self) -> dict:
        sg = self.signed()
        signs = [d["sign"] for _, _, d in sg.edges(data=True)]
        return {
            "relations": len(self.relations),
            "entities": sg.number_of_nodes(),
            "signed_edges": sg.number_of_edges(),
            "positive_edges": sum(1 for s in signs if s > 0),
            "negative_edges": sum(1 for s in signs if s < 0),
            "contested_edges": sum(1 for _, _, d in sg.edges(data=True) if d["contested"]),
        }

    # -------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entity_types": self.entity_types,
            "relations": [r.to_dict() for r in self.relations],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SignedKnowledgeGraph:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kg = cls()
        kg.entity_types = data.get("entity_types", {})
        kg.add_many(SignedRelation(**r) for r in data["relations"])
        return kg
