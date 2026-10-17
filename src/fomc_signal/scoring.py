"""Score FOMC statements with an LLM.

Two complementary measures:

* **Document assessment**: one call per statement returns a stance score in
  [-1, 1] plus structured fields (policy action, guidance, concern levels)
  and up to four supporting quotes. The quotes are checked against the
  statement and the call is retried with feedback if any of them isn't a
  verbatim extract, which catches the model paraphrasing or inventing
  evidence.
* **Sentence classification**: every policy sentence is labelled hawkish,
  dovish or neutral in one batched call, using the same definitions as the
  human annotators of the Trillion Dollar Words dataset (Shah, Paturi and
  Chava, 2023). Net tone = (hawkish - dovish) / sentences. Because the
  labels are per sentence, this measure can be validated directly against
  human labels.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .llm import LLMError, Provider
from .text import policy_sentences

DEFINITIONS = """Definitions (from the perspective of US monetary policy):
- hawkish: points towards tighter policy or higher rates in future, e.g. concern about high or rising inflation, a strong economy or labor market that could overheat, rate increases, shrinking the balance sheet, or removing accommodation.
- dovish: points towards easier policy or lower rates, e.g. weak or slowing growth, rising unemployment, downside risks, inflation running low or falling, rate cuts, asset purchases, or keeping policy accommodative.
- neutral: neither, e.g. purely descriptive statements, procedural text, or balanced statements with no clear direction."""

DOC_SYSTEM = ("You are an experienced Fed watcher on a macro hedge fund desk. You read FOMC statements "
              "and judge how hawkish or dovish they are, relative to a neutral policy stance.\n\n" + DEFINITIONS)

DOC_SCHEMA = {
    "type": "object",
    "required": ["stance", "policy_action", "forward_guidance", "inflation_concern", "growth_concern",
                 "key_phrases", "rationale"],
    "properties": {
        "stance": {"type": "number", "minimum": -1, "maximum": 1,
                   "description": "-1 very dovish, 0 neutral, +1 very hawkish"},
        "policy_action": {"type": "string", "enum": ["hike", "cut", "hold", "other"]},
        "forward_guidance": {"type": "string", "enum": ["tightening", "easing", "neutral", "none"]},
        "inflation_concern": {"type": "integer", "minimum": 1, "maximum": 5},
        "growth_concern": {"type": "integer", "minimum": 1, "maximum": 5},
        "key_phrases": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4,
                        "description": "verbatim quotes from the statement that drive the assessment"},
        "rationale": {"type": "string", "description": "at most 60 words"},
    },
}

SENT_SYSTEM = "You label sentences from FOMC statements as hawkish, dovish or neutral.\n\n" + DEFINITIONS

SENT_SCHEMA = {
    "type": "object",
    "required": ["labels"],
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["i", "label"],
                "properties": {"i": {"type": "integer"}, "label": {"type": "string", "enum": ["hawkish", "dovish", "neutral"]}},
            },
        }
    },
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def unverified_quotes(quotes: list[str], text: str) -> list[str]:
    body = _norm(text)
    return [q for q in quotes if _norm(q) not in body]


def assess_statement(provider: Provider, text: str, date: str, run: int = 0, max_fix: int = 2) -> dict:
    user = f"FOMC statement released {date}:\n\n<statement>\n{text}\n</statement>\n\nAssess this statement."
    out = provider.complete_json(DOC_SYSTEM, user, DOC_SCHEMA, run=run)
    for _ in range(max_fix):
        bad = unverified_quotes(out["key_phrases"], text)
        if not bad:
            break
        fix = (user + "\n\nThese key_phrases are not verbatim quotes from the statement: "
               + "; ".join(f'"{b}"' for b in bad) + ". Use exact extracts only.")
        out = provider.complete_json(DOC_SYSTEM, fix, DOC_SCHEMA, run=run)
    out = dict(out)
    out["quotes_verified"] = not unverified_quotes(out["key_phrases"], text)
    return out


def label_sentences(provider: Provider, sentences: list[str], run: int = 0) -> list[str]:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    user = (f"Label each of the {len(sentences)} numbered sentences. Return one entry per sentence, "
            f"using its number as i.\n\n{numbered}")
    out = provider.complete_json(SENT_SYSTEM, user, SENT_SCHEMA, run=run)
    labels = {int(d["i"]): d["label"] for d in out["labels"]}
    missing = [i for i in range(len(sentences)) if i not in labels]
    if missing:
        raise LLMError(f"missing labels for sentences {missing[:5]}")
    return [labels[i] for i in range(len(sentences))]


def net_tone(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return (labels.count("hawkish") - labels.count("dovish")) / len(labels)


def score_statements(statements: pd.DataFrame, provider: Provider, runs: int = 1, progress: bool = True) -> pd.DataFrame:
    """Document and sentence scores for each statement (columns prefixed llm_)."""
    rows = []
    for k, row in enumerate(statements.itertuples(index=False)):
        date = pd.Timestamp(row.date).strftime("%Y-%m-%d")
        sents = policy_sentences(row.text)
        rec = {"date": pd.Timestamp(row.date)}
        stances = []
        for run in range(runs):
            doc = assess_statement(provider, row.text, date, run=run)
            stances.append(doc["stance"])
            if run == 0:
                rec.update({f"llm_{key}": doc[key] for key in
                            ("stance", "policy_action", "forward_guidance", "inflation_concern", "growth_concern",
                             "quotes_verified", "rationale")})
                rec["llm_key_phrases"] = " | ".join(doc["key_phrases"])
        if runs > 1:
            rec["llm_stance_std"] = float(np.std(stances, ddof=1))
        labels = label_sentences(provider, sents)
        rec["llm_net_tone"] = net_tone(labels)
        rec["n_sentences"] = len(sents)
        rows.append(rec)
        if progress and (k + 1) % 10 == 0:
            u = provider.usage
            print(f"  {k + 1}/{len(statements)} statements, {u.calls} calls ({u.cached} cached), ${u.cost_usd:.2f}")
    return pd.DataFrame(rows)
