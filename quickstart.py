"""Minimal end-to-end example using the Python API (no API key needed)."""

from pathlib import Path

from signed_graphrag import (
    RuleBasedExtractor,
    SignedKnowledgeGraph,
    SignedRetriever,
    analyze_balance,
    signed_communities,
)
from signed_graphrag.data_io import load_articles, load_entities

DATA = Path(__file__).resolve().parents[1] / "data"

entities = load_entities(DATA / "entities.json")
articles = load_articles(DATA / "sample_news.jsonl")

# 1. Signed extraction
kg = SignedKnowledgeGraph()
kg.add_many(RuleBasedExtractor(entities).extract_many(articles))
g = kg.signed()
print(kg.summary())

# 2. Balance checking
report = analyze_balance(g)
print(f"\nBalance ratio: {report.balance_ratio:.0%}")
for tri in report.unbalanced:
    print("  unbalanced:", tri.describe())

# 3. Factions
factions = signed_communities(g)
print("\nFactions:", factions)

# 4. Signed retrieval context for a question
print("\n" + SignedRetriever(g, entities).build_context("Is Kestria an ally of the Green Union?"))
