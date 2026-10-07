"""Loading news articles and entity gazetteers."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Article, Entity


def load_articles(path: str | Path) -> list[Article]:
    """Load articles from a .jsonl file (one JSON object per line) with keys
    id, title, text and optionally date, source, url."""
    articles = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                articles.append(Article(**{k: d.get(k, "") for k in
                                           ("id", "title", "text", "date", "source", "url")}))
    return articles


def load_entities(path: str | Path) -> list[Entity]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Entity(name=e["name"], type=e.get("type", "OTHER"), aliases=e.get("aliases", []))
            for e in data]


def fetch_rss(feed_urls: list[str], limit_per_feed: int = 20) -> list[Article]:
    """Fetch real news from RSS feeds (requires `pip install feedparser`).

    RSS items usually contain only a headline and summary. For full text,
    combine with a scraper such as `trafilatura`, and respect each
    publisher's terms of use.
    """
    import feedparser  # optional dependency

    articles = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        source = feed.feed.get("title", url)
        for i, item in enumerate(feed.entries[:limit_per_feed]):
            articles.append(Article(
                id=f"{abs(hash(url)) % 10_000}-{i}",
                title=item.get("title", ""),
                text=item.get("summary", ""),
                date=item.get("published", ""),
                source=source,
                url=item.get("link", ""),
            ))
    return articles


def save_articles(articles: list[Article], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(a.__dict__, ensure_ascii=False) + "\n" for a in articles)
