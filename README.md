# Signed GraphRAG for news and politics

GraphRAG builds a knowledge graph from documents and retrieves over it. Standard GraphRAG only records *that* two entities are connected. **Signed GraphRAG** also records *how*. Each edge is **positive (+)** (supports, allies with, endorses) or **negative (−)** (opposes, criticizes, sanctions).

In politics, that sign is often the whole story. "Party A" and "Party B" being connected means very different things if they are coalition partners or bitter rivals.

![Signed graph of the sample dataset](docs/signed_graph.png)

## What this project does

| Stage | Module | What happens |
|---|---|---|
| 1. Signed extraction | `extraction.py` | Pulls `(source, relation, target, sign, confidence, evidence)` from news text. It has a **rule-based** extractor (offline, no API key) and an **LLM** extractor (Claude). Both handle negation ("refused to endorse" → −) and hedging ("reportedly" lowers confidence). |
| 2. Signed graph | `graph.py` | Merges all relations per entity pair into one signed edge. Pairs with both positive and negative evidence are marked **contested**. Ties go to the most recent relation. |
| 3. Balance checking | `balance.py` | Finds every triangle and checks **structural balance** (product of signs > 0). Unbalanced triangles flag contradictions, shifting alliances or extraction errors. It also finds the best **two-camp split** and the edges that violate it (frustration). |
| 4. Faction detection | `communities.py` | Signed correlation clustering: positive edges inside communities, negative edges between them. This replaces Leiden, which cannot use negative weights. |
| 5. Signed retrieval | `retrieval.py` | Finds entities in a question, walks **signed paths** between them (sign of path = product of edge signs), flags disagreeing paths and local imbalance, and builds an evidence-cited context for the LLM. |

### Structural balance in one picture

```
Balanced (stable)                    Unbalanced (tension)
  A --+-- B      A --+-- B             A --+-- B       A --−-- B
   \     /        \     /               \     /         \     /
    +   +          −   −                 +   −           −   −
     \ /            \ /                   \ /             \ /
      C              C                     C               C
 friend of friend   enemy of enemy     two of my friends   all enemies
   = friend          = friend            are enemies
```

## Quick start

```bash
git clone https://github.com/<your-username>/signed-graphrag.git
cd signed-graphrag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 1. Extract signed relations and build the graph (offline, rule-based)
signed-graphrag build --articles data/sample_news.jsonl --entities data/entities.json

# 2. Balance check + factions + plot  (writes to outputs/)
signed-graphrag analyze

# 3. Ask a question (prints the signed retrieval context)
signed-graphrag query --entities data/entities.json "How is Senator Kovac connected to Prime Minister Vale?"

# Tests
pytest -q
```

### Example output

```
Triangles: 13   balance ratio: 76.92%
Unbalanced triangles (check these):
   Ada Kovac (-) Elena Vale (+) National Alliance (+) Ada Kovac
   ...
Best two-camp split (frustration = 2 edges):
  Camp A: Aldermoor, Elena Vale, Jonas Reed, Liberty Party, National Alliance, Northgate Energy
  Camp B: Ada Kovac, Carbon Tax Bill, Green Union, Kestria, Marcus Thorne, Priya Desai, ...
```

The Kovac triangle is deliberate in the sample data. Ada Kovac belongs to the ruling National Alliance, which backs Prime Minister Vale, but Kovac publicly criticized Vale. The system flags this as an internal party rift. When you query that pair, the retriever reports that the direct edge is negative while the path through the party is positive, and warns that the paths disagree.

## Using an LLM

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...          # see .env.example

signed-graphrag build --extractor llm --articles data/sample_news.jsonl --entities data/entities.json
signed-graphrag query --llm "Which parties oppose the Carbon Tax Bill, and is the opposition united?"
```

The model defaults to `claude-sonnet-5-5` and can be changed with `--model` or `SIGNED_GRAPHRAG_MODEL`. To use another provider, override `LLMExtractor._call_llm`.

## Using real news

The sample dataset (`data/sample_news.jsonl`) is **synthetic**. It describes a fictional country, Veloria, with made-up politicians, parties and countries, so the repo has no claims about real people. To use real news:

1. **Get articles** into JSONL (`id`, `title`, `text`, `date`, `source`, `url`). Options:
   - RSS feeds: `signed_graphrag.data_io.fetch_rss([...])` (needs `pip install -e ".[rss]"`)
   - [GDELT](https://www.gdeltproject.org/): a free global news event database, which already includes CAMEO conflict/cooperation codes that map naturally to signs
   - News APIs (NewsAPI, The Guardian Open Platform, etc.)
2. **Use the LLM extractor.** Real news is too varied for the rule-based lexicon.
3. Optionally provide an `entities.json` with canonical names and aliases. This greatly reduces duplicate nodes ("PM", "the Prime Minister", full name).

Respect publishers' terms of use when collecting article text.

## Python API

```python
from signed_graphrag import (RuleBasedExtractor, SignedKnowledgeGraph,
                             SignedRetriever, analyze_balance, signed_communities)
from signed_graphrag.data_io import load_articles, load_entities

entities = load_entities("data/entities.json")
kg = SignedKnowledgeGraph()
kg.add_many(RuleBasedExtractor(entities).extract_many(load_articles("data/sample_news.jsonl")))

g = kg.signed()
report = analyze_balance(g)              # triangles, balance ratio, camps, suspicious edges
factions = signed_communities(g)         # node -> faction id
context = SignedRetriever(g, entities).build_context("Is Kestria an ally of the Green Union?")
```

See `examples/quickstart.py`.

## Project layout

```
signed_graphrag/
  schema.py        Article, Entity, SignedRelation
  extraction.py    RuleBasedExtractor, LLMExtractor, polarity lexicon
  graph.py         SignedKnowledgeGraph (aggregation, contested edges, save/load)
  balance.py       triangles, structural balance, two-camp partition, frustration
  communities.py   signed correlation clustering, faction summaries
  retrieval.py     signed path search, context building, LLM answering
  visualize.py     matplotlib plot
  data_io.py       JSONL / RSS loading
  cli.py           build | analyze | query
data/              synthetic sample news + entity gazetteer
tests/             pytest suite
examples/          quickstart script
```

## Limitations and ideas

- **Sign errors propagate.** A wrong sign flips path conclusions. Check `suspicious_edges` in the balance report, which lists edges involved in the most unbalanced triangles.
- **Sign composition is a heuristic.** "Enemy of my enemy" is reasoning from balance theory, not a fact. The retriever labels these as *implied* and lowers their confidence with path length.
- **Time matters.** Alliances change. Edges keep dates, and ties go to the latest relation, but a full temporal graph (edge signs per time window) is a natural next step.
- **Rule-based extraction is a baseline only.** It links consecutive entity mentions in a sentence and misses coordination ("A and B oppose C") and pronouns.
- Next steps: LLM-written community reports, global search over faction summaries, signed embeddings (e.g. SiGAT, SGCN), and a temporal balance timeline.

## License

MIT, see [LICENSE](LICENSE).
