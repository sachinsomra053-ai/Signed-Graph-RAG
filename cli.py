"""Command line interface.

    python -m signed_graphrag build   --articles data/sample_news.jsonl --entities data/entities.json
    python -m signed_graphrag analyze --graph outputs/graph.json
    python -m signed_graphrag query   --graph outputs/graph.json "Is Aldermoor an ally of the Green Union?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .balance import analyze_balance
from .communities import community_stats, signed_communities, summarize_community
from .data_io import load_articles, load_entities
from .extraction import LLMExtractor, RuleBasedExtractor
from .graph import SignedKnowledgeGraph
from .retrieval import SignedRetriever


def cmd_build(args) -> None:
    articles = load_articles(args.articles)
    entities = load_entities(args.entities) if args.entities else []
    if args.extractor == "llm":
        extractor = LLMExtractor(model=args.model, known_entities=entities)
    else:
        if not entities:
            raise SystemExit("The rule-based extractor needs --entities (a gazetteer).")
        extractor = RuleBasedExtractor(entities)

    kg = SignedKnowledgeGraph()
    kg.set_entity_types({e.name: e.type for e in entities})
    for art in articles:
        rels = extractor.extract(art)
        kg.add_many(rels)
        print(f"[{art.id}] {len(rels):2d} relations  {art.title[:60]}")
    kg.save(args.out)
    print("\nGraph summary:", json.dumps(kg.summary(), indent=2))
    print(f"Saved to {args.out}")


def cmd_analyze(args) -> None:
    kg = SignedKnowledgeGraph.load(args.graph)
    g = kg.signed()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = analyze_balance(g)
    print(f"Triangles: {len(report.triangles)}   balance ratio: {report.balance_ratio:.2%}")
    print("Patterns:", report.pattern_counts)
    if report.unbalanced:
        print("\nUnbalanced triangles (check these):")
        for t in report.unbalanced:
            print("  ", t.describe())
    camp = {0: [], 1: []}
    for n, c in report.camps.items():
        camp[c].append(n)
    print(f"\nBest two-camp split (frustration = {report.frustration} edges):")
    print("  Camp A:", ", ".join(sorted(camp[0])))
    print("  Camp B:", ", ".join(sorted(camp[1])))
    if report.suspicious_edges:
        print("\nEdges most involved in imbalance (review sign / evidence):")
        for u, v, s in report.suspicious_edges[:5]:
            print(f"   {u} {'+' if s > 0 else '-'} {v}")

    comm = signed_communities(g)
    stats = community_stats(g, comm)
    print("\nSigned communities (factions):")
    reports = {}
    for s in stats:
        print(f"  #{s['community']}: {', '.join(s['members'])}  "
              f"(in +{s['internal_positive']}/-{s['internal_negative']}, "
              f"out +{s['external_positive']}/-{s['external_negative']})")
        reports[s["community"]] = summarize_community(g, s["members"], comm)

    (out / "balance_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (out / "communities.json").write_text(json.dumps(
        {"assignment": comm, "stats": stats, "reports": reports}, indent=2), encoding="utf-8")
    if not args.no_plot:
        from .visualize import draw_signed_graph
        draw_signed_graph(g, comm, out / "signed_graph.png")
        print(f"\nPlot saved to {out / 'signed_graph.png'}")
    print(f"Reports saved to {out}/")


def cmd_query(args) -> None:
    kg = SignedKnowledgeGraph.load(args.graph)
    entities = load_entities(args.entities) if args.entities else []
    retriever = SignedRetriever(kg.signed(), entities)
    if args.llm:
        print(retriever.answer(args.question, model=args.model))
    else:
        print(retriever.build_context(args.question))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="signed-graphrag",
                                description="Signed GraphRAG for news and politics")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="extract signed relations and build the graph")
    b.add_argument("--articles", required=True)
    b.add_argument("--entities", default=None)
    b.add_argument("--extractor", choices=["rules", "llm"], default="rules")
    b.add_argument("--model", default=None)
    b.add_argument("--out", default="outputs/graph.json")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("analyze", help="balance check, factions, plot")
    a.add_argument("--graph", default="outputs/graph.json")
    a.add_argument("--out-dir", default="outputs")
    a.add_argument("--no-plot", action="store_true")
    a.set_defaults(func=cmd_analyze)

    q = sub.add_parser("query", help="signed local search")
    q.add_argument("question")
    q.add_argument("--graph", default="outputs/graph.json")
    q.add_argument("--entities", default=None)
    q.add_argument("--llm", action="store_true", help="generate an answer with the LLM")
    q.add_argument("--model", default=None)
    q.set_defaults(func=cmd_query)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
