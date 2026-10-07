import json

import pytest

from signed_graphrag.extraction import LLMExtractor, RuleBasedExtractor
from signed_graphrag.schema import Article, Entity, SignedRelation

ENTS = [Entity("Green Union", "PARTY"), Entity("Liberty Party", "PARTY"),
        Entity("Elena Vale", "PERSON", ["Vale"]), Entity("Carbon Tax Bill", "POLICY")]


def run(text):
    return RuleBasedExtractor(ENTS).extract(Article(id="t", title="", text=text))


def test_positive_relation():
    (r,) = run("The Liberty Party allied with the Green Union on press freedom.")
    assert (r.source, r.target, r.sign) == ("Liberty Party", "Green Union", 1)


def test_negative_relation():
    (r,) = run("Elena Vale attacked the Green Union over energy prices.")
    assert r.sign == -1 and r.relation == "attacks"


def test_negation_flips_sign():
    (r,) = run("Vale refused to endorse the Carbon Tax Bill.")
    assert r.sign == -1 and r.relation.startswith("not_")


def test_hedge_lowers_confidence():
    (plain,) = run("Elena Vale backs the Carbon Tax Bill.")
    (hedged,) = run("Elena Vale reportedly backs the Carbon Tax Bill.")
    assert hedged.confidence < plain.confidence


def test_neutral_relation():
    (r,) = run("Elena Vale met with the Green Union leadership.")
    assert r.sign == 0


def test_invalid_sign_rejected():
    with pytest.raises(ValueError):
        SignedRelation("a", "b", "x", sign=2)


def test_llm_response_parsing():
    raw = "Here you go:\n" + json.dumps({"entities": [], "relations": [
        {"source": "A", "target": "B", "relation": "opposes", "sign": -1, "confidence": 0.9}]})
    data = LLMExtractor.parse_response(raw)
    assert data["relations"][0]["sign"] == -1
    assert LLMExtractor.parse_response("no json") == {"entities": [], "relations": []}


def test_llm_extractor_with_fake_model(monkeypatch):
    ext = LLMExtractor()
    fake = json.dumps({"relations": [
        {"source": "Green Union", "target": "Elena Vale", "relation": "criticizes", "sign": -1},
        {"source": "x", "target": "y", "relation": "bad", "sign": 5},
    ]})
    monkeypatch.setattr(ext, "_call_llm", lambda prompt: fake)
    rels = ext.extract(Article(id="a1", title="t", text="..."))
    assert len(rels) == 1 and rels[0].sign == -1 and rels[0].extractor == "llm"
