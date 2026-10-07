"""Core data structures for signed relations extracted from news text."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Sign = Literal[1, -1, 0]

ENTITY_TYPES = ("PERSON", "PARTY", "COUNTRY", "ORGANIZATION", "POLICY", "OTHER")


@dataclass
class Article:
    """A single news article."""

    id: str
    title: str
    text: str
    date: str = ""
    source: str = ""
    url: str = ""


@dataclass
class Entity:
    name: str
    type: str = "OTHER"
    aliases: list[str] = field(default_factory=list)


@dataclass
class SignedRelation:
    """A relationship with polarity.

    sign:        +1 positive (supports, allies with, endorses ...)
                 -1 negative (opposes, criticizes, sanctions ...)
                  0 neutral  (meets, visits, located in ...)
    confidence:  0..1, how sure the extractor is
    """

    source: str
    target: str
    relation: str
    sign: int
    confidence: float = 1.0
    evidence: str = ""
    article_id: str = ""
    date: str = ""
    extractor: str = ""

    def __post_init__(self) -> None:
        if self.sign not in (1, -1, 0):
            raise ValueError(f"sign must be +1, -1 or 0, got {self.sign!r}")
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def weight(self) -> float:
        """Signed weight used when aggregating edges."""
        return self.sign * self.confidence

    def to_dict(self) -> dict:
        return asdict(self)
