from pathlib import Path

from signed_graphrag import (
    RuleBasedExtractor,
    SignedKnowledgeGraph,
    SignedRelation,
    SignedRetriever,
    analyze_balance,
    signed_communities,
)
from signed_graphrag.data_io import load_articles, load_entities

DATA = Path(__file__).resolve().parents[1] / "data"


def build():
    ents = load_entities(DATA / "entities.json")
    kg = SignedKnowledgeGraph()
    kg.add_many(RuleBasedExtractor(ents).extract_many(load_articles(DATA / "sample_news.jsonl")))
    return kg, ents


def test_sample_pipeline_end_to_end():
    kg, ents = build()
    s = kg.summary()
    assert s["positive_edges"] > 0 and s["negative_edges"] > 0
    g = kg.signed()
    rep = analyze_balance(g)
    assert 0 < rep.balance_ratio < 1  # sample data contains deliberate contradictions
    comm = signed_communities(g)
    assert comm["Elena Vale"] == comm["National Alliance"]
    assert comm["Elena Vale"] != comm["Green Union"]
    ctx = SignedRetriever(g, ents).build_context("How is Senator Kovac linked to Vale?")
    assert "Ada Kovac" in ctx and "Elena Vale" in ctx and "Path:" in ctx


def test_contested_edge_uses_latest_sign():
    kg = SignedKnowledgeGraph()
    kg.add(SignedRelation("A", "B", "allied_with", 1, 0.8, date="2026-01-01"))
    kg.add(SignedRelation("A", "B", "broke_with", -1, 0.8, date="2026-02-01"))
    d = kg.signed()["A"]["B"]
    assert d["contested"] and d["sign"] == -1


def test_save_and_load(tmp_path):
    kg, _ = build()
    kg.save(tmp_path / "g.json")
    kg2 = SignedKnowledgeGraph.load(tmp_path / "g.json")
    assert kg2.summary() == kg.summary()
