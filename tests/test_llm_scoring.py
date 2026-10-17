from dataclasses import dataclass, field

import pytest

from fomc_signal.llm import LLMError, Provider, extract_json, validate
from fomc_signal.scoring import DOC_SCHEMA, SENT_SCHEMA, assess_statement, label_sentences, net_tone, unverified_quotes

STATEMENT = ("Inflation remains elevated. The Committee decided to raise the target range for the federal funds "
             "rate to 5-1/4 to 5-1/2 percent.")


@dataclass
class FakeProvider(Provider):
    """Returns scripted replies in order and records the prompts it saw."""

    replies: list = field(default_factory=list)
    prompts: list = field(default_factory=list)
    model: str = "fake"
    name: str = "fake"

    def _call(self, system, user, schema, temperature):
        self.prompts.append(user)
        return self.replies.pop(0), 100, 20


def doc_reply(quotes, stance=0.6):
    return {"stance": stance, "policy_action": "hike", "forward_guidance": "tightening", "inflation_concern": 4,
            "growth_concern": 2, "key_phrases": quotes, "rationale": "Rate hike with elevated inflation."}


def test_validate_catches_schema_violations():
    bad = doc_reply(["x"], stance=1.7)
    bad["policy_action"] = "pause"
    errs = validate(bad, DOC_SCHEMA)
    assert any("maximum" in e for e in errs) and any("pause" in e for e in errs)
    assert validate(doc_reply(["x"]), DOC_SCHEMA) == []


def test_extract_json_from_fenced_reply():
    assert extract_json('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_invalid_reply_is_retried_with_feedback(tmp_path):
    p = FakeProvider(cache_dir=tmp_path, replies=[{"stance": 3}, doc_reply(["Inflation remains elevated."])])
    out = p.complete_json("sys", "user", DOC_SCHEMA)
    assert out["stance"] == 0.6
    assert "previous reply was invalid" in p.prompts[1]


def test_results_are_cached(tmp_path):
    p = FakeProvider(cache_dir=tmp_path, replies=[doc_reply(["Inflation remains elevated."])])
    first = p.complete_json("sys", "user", DOC_SCHEMA)
    second = p.complete_json("sys", "user", DOC_SCHEMA)  # would fail if it called the model again
    assert first == second and p.usage.cached == 1 and p.usage.calls == 1


def test_unverified_quotes_tolerates_whitespace_and_case():
    assert unverified_quotes(["inflation  REMAINS elevated"], STATEMENT) == []
    assert unverified_quotes(["inflation is running hot"], STATEMENT) == ["inflation is running hot"]


def test_hallucinated_quote_triggers_correction(tmp_path):
    p = FakeProvider(cache_dir=tmp_path, replies=[doc_reply(["inflation is running hot"]),
                                                  doc_reply(["Inflation remains elevated."])])
    out = assess_statement(p, STATEMENT, "2023-07-26")
    assert out["quotes_verified"] is True
    assert "not verbatim quotes" in p.prompts[1]


def test_label_sentences_and_net_tone(tmp_path):
    reply = {"labels": [{"i": 0, "label": "hawkish"}, {"i": 1, "label": "neutral"}]}
    p = FakeProvider(cache_dir=tmp_path, replies=[reply])
    labels = label_sentences(p, ["Inflation remains elevated.", "The Committee met."])
    assert labels == ["hawkish", "neutral"]
    assert net_tone(labels) == 0.5
    assert validate(reply, SENT_SCHEMA) == []


def test_missing_sentence_label_raises(tmp_path):
    p = FakeProvider(cache_dir=tmp_path, replies=[{"labels": [{"i": 0, "label": "dovish"}]}])
    with pytest.raises(LLMError):
        label_sentences(p, ["a b c", "d e f"])
