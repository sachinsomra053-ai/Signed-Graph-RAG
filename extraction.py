"""Signed relation extraction.

Two extractors share one interface (`extract(article) -> list[SignedRelation]`):

* RuleBasedExtractor - offline, no API key. Uses an entity gazetteer plus a
  polarity lexicon of political verbs. Good for demos, tests and as a baseline.
* LLMExtractor       - asks an LLM (Anthropic Claude by default) to return
  entities and signed relations as JSON. Much better recall and handles
  negation, hedging and indirect phrasing.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable

from .schema import Article, Entity, SignedRelation

# ---------------------------------------------------------------------------
# Polarity lexicon for news & politics. Keys are regex fragments (matched on
# lowercased text), values are (canonical relation, sign).
# ---------------------------------------------------------------------------
POLARITY_LEXICON: dict[str, tuple[str, int]] = {
    # positive
    r"endors(?:e|es|ed|ing)": ("endorses", 1),
    r"support(?:s|ed|ing)?": ("supports", 1),
    r"back(?:s|ed|ing)": ("backs", 1),
    r"allie[sd] with|ally with|allying with": ("allied_with", 1),
    r"form(?:s|ed)? (?:a )?coalition with|join(?:s|ed)? (?:a )?coalition with": ("coalition_with", 1),
    r"praise[sd]?|prais(?:ing)": ("praises", 1),
    r"sign(?:s|ed)? (?:an? )?(?:\w+ )?(?:deal|agreement|pact|treaty) with": ("agreement_with", 1),
    r"leads|led|member of|joined": ("affiliated_with", 1),
    r"partner(?:s|ed)? with": ("partners_with", 1),
    r"cooperat(?:e|es|ed|ing) with": ("cooperates_with", 1),
    r"welcome[sd]?": ("welcomes", 1),
    r"defend(?:s|ed)?": ("defends", 1),
    r"vot(?:e|es|ed) for": ("votes_for", 1),
    # negative
    r"oppos(?:e|es|ed|ing)": ("opposes", -1),
    r"criticiz(?:e|es|ed|ing)|criticis(?:e|es|ed|ing)": ("criticizes", -1),
    r"condemn(?:s|ed|ing)?": ("condemns", -1),
    r"accus(?:e|es|ed|ing)": ("accuses", -1),
    r"sanction(?:s|ed|ing)?": ("sanctions", -1),
    r"attack(?:s|ed|ing)?": ("attacks", -1),
    r"su(?:e|es|ed|ing)": ("sues", -1),
    r"reject(?:s|ed|ing)?": ("rejects", -1),
    r"denounc(?:e|es|ed|ing)": ("denounces", -1),
    r"blame[sd]?|blaming": ("blames", -1),
    r"impos(?:e|es|ed) tariffs on": ("tariffs_on", -1),
    r"expel(?:s|led)?": ("expels", -1),
    r"vot(?:e|es|ed) against": ("votes_against", -1),
    r"(?:quit|quits|left|leaves) (?:the )?coalition with|broke with|breaks with": ("broke_with", -1),
    r"clash(?:es|ed)? with": ("clashes_with", -1),
    r"rival(?:s)?": ("rival_of", -1),
    # neutral (kept so they are not mistaken for signed edges)
    r"met with|meets with|meet with": ("met_with", 0),
    r"visit(?:s|ed)?": ("visits", 0),
    r"spoke with|speaks with|talk(?:s|ed) with": ("talked_with", 0),
}

NEGATIONS = re.compile(
    r"\b(?:not|no longer|never|refus(?:e|es|ed) to|declin(?:e|es|ed) to|did not|does not|won't|will not)\b"
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
HEDGES = re.compile(r"\b(?:may|might|could|reportedly|allegedly|rumou?red)\b")


class Extractor(ABC):
    name = "base"

    @abstractmethod
    def extract(self, article: Article) -> list[SignedRelation]:
        ...

    def extract_many(self, articles: Iterable[Article]) -> list[SignedRelation]:
        out: list[SignedRelation] = []
        for art in articles:
            out.extend(self.extract(art))
        return out


class RuleBasedExtractor(Extractor):
    """Gazetteer + polarity-lexicon extractor (subject - verb - object)."""

    name = "rules"

    def __init__(self, entities: list[Entity]):
        self.entities = entities
        # alias -> canonical name, longest aliases first so "Prime Minister Vale"
        # wins over "Vale".
        pairs = []
        for e in entities:
            for alias in {e.name, *e.aliases}:
                pairs.append((alias, e.name))
        pairs.sort(key=lambda p: -len(p[0]))
        self._alias_re = re.compile(
            "|".join(rf"\b{re.escape(a)}\b" for a, _ in pairs)
        ) if pairs else None
        self._alias_map = {a: n for a, n in pairs}
        self._verbs = [(re.compile(rf"\b(?:{pat})\b"), rel, sign)
                       for pat, (rel, sign) in POLARITY_LEXICON.items()]

    def _find_entities(self, sentence: str) -> list[tuple[int, int, str]]:
        if not self._alias_re:
            return []
        return [(m.start(), m.end(), self._alias_map[m.group(0)])
                for m in self._alias_re.finditer(sentence)]

    def extract(self, article: Article) -> list[SignedRelation]:
        relations: list[SignedRelation] = []
        text = f"{article.title}. {article.text}"
        for sentence in SENTENCE_SPLIT.split(text):
            ents = self._find_entities(sentence)
            if len(ents) < 2:
                continue
            lower = sentence.lower()
            # consecutive entity pairs: subject ... verb ... object
            for (s0, s1, subj), (o0, o1, obj) in zip(ents, ents[1:]):
                if subj == obj:
                    continue
                span = lower[s1:o0]
                for verb_re, rel, sign in self._verbs:
                    m = verb_re.search(span)
                    if not m:
                        continue
                    final_sign, conf = sign, 0.8
                    # negation directly before the verb flips polarity
                    if sign != 0 and NEGATIONS.search(span[: m.start()][-30:]):
                        final_sign, rel = -sign, f"not_{rel}"
                        conf = 0.6
                    if HEDGES.search(span):
                        conf *= 0.6
                    relations.append(SignedRelation(
                        source=subj, target=obj, relation=rel, sign=final_sign,
                        confidence=conf, evidence=sentence.strip(),
                        article_id=article.id, date=article.date,
                        extractor=self.name,
                    ))
                    break
        return relations


# ---------------------------------------------------------------------------
# LLM extractor
# ---------------------------------------------------------------------------
LLM_PROMPT = """You extract SIGNED relationships from political news.

Return ONLY a JSON object of the form:
{{
  "entities": [{{"name": "...", "type": "PERSON|PARTY|COUNTRY|ORGANIZATION|POLICY|OTHER"}}],
  "relations": [
    {{"source": "...", "target": "...", "relation": "short_snake_case_verb",
      "sign": 1 | -1 | 0, "confidence": 0.0-1.0,
      "evidence": "exact sentence from the article"}}
  ]
}}

Sign rules:
 +1  cooperation or approval: supports, endorses, allies with, signs deal with, praises, votes for
 -1  conflict or disapproval: opposes, criticizes, sanctions, sues, accuses, votes against, breaks with
  0  neutral contact or fact: meets, visits, is member of, is located in
Handle negation ("refused to endorse" is -1) and lower confidence for hedged or
reported claims ("reportedly", "may"). Use canonical full names for entities and
reuse the same spelling every time. Only include relations stated in the text.
{known}
ARTICLE (id={id}, date={date}):
{title}

{text}
"""


class LLMExtractor(Extractor):
    """Extract signed relations with an LLM. Requires `pip install anthropic`
    and ANTHROPIC_API_KEY. Swap `_call_llm` to use any other provider."""

    name = "llm"

    def __init__(self, model: str | None = None, known_entities: list[Entity] | None = None,
                 max_tokens: int = 2000):
        self.model = model or os.getenv("SIGNED_GRAPHRAG_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self.known_entities = known_entities or []
        self._client = None

    def _call_llm(self, prompt: str) -> str:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install the 'anthropic' package to use LLMExtractor") from exc
            self._client = anthropic.Anthropic()
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    @staticmethod
    def parse_response(raw: str) -> dict:
        """Pull the first JSON object out of the model output."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {"entities": [], "relations": []}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}

    def extract(self, article: Article) -> list[SignedRelation]:
        known = ""
        if self.known_entities:
            names = ", ".join(e.name for e in self.known_entities)
            known = f"Prefer these canonical entity names when they apply: {names}\n"
        prompt = LLM_PROMPT.format(known=known, id=article.id, date=article.date,
                                   title=article.title, text=article.text)
        data = self.parse_response(self._call_llm(prompt))
        out: list[SignedRelation] = []
        for r in data.get("relations", []):
            try:
                sign = int(r.get("sign", 0))
                if sign not in (1, -1, 0) or not r.get("source") or not r.get("target"):
                    continue
                out.append(SignedRelation(
                    source=r["source"].strip(), target=r["target"].strip(),
                    relation=str(r.get("relation", "related_to")), sign=sign,
                    confidence=float(r.get("confidence", 0.7)),
                    evidence=str(r.get("evidence", "")),
                    article_id=article.id, date=article.date, extractor=self.name,
                ))
            except (TypeError, ValueError):
                continue
        return out
