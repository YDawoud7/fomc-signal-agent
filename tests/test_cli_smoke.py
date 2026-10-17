"""End-to-end run of score + analyze with a fake LLM and synthetic market data."""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fomc_signal import cli
from fomc_signal.llm import Provider


@dataclass
class KeywordProvider(Provider):
    """Deterministic stand-in for an LLM: hawkish if the text mentions raising rates."""

    model: str = "fake"
    name: str = "fake"

    def _call(self, system, user, schema, temperature):
        if "labels" in schema["properties"]:
            lines = [l for l in user.splitlines() if l[:1].isdigit()]
            labs = [{"i": int(l.split(".")[0]), "label": "hawkish" if "raise" in l else "dovish" if "lower" in l else "neutral"}
                    for l in lines]
            return {"labels": labs}, 50, 10
        body = user.split("<statement>")[1]
        hawk = "raise" in body
        quote = body.strip().split(".")[0] + "."
        return {"stance": 0.5 if hawk else -0.5, "policy_action": "hold", "forward_guidance": "none",
                "inflation_concern": 3, "growth_concern": 3, "key_phrases": [quote], "rationale": "test"}, 100, 20


def test_score_and_analyze_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "DATA", tmp_path / "data")
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cli, "FIG", tmp_path / "figures")
    (tmp_path / "data" / "fred").mkdir(parents=True)

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010-01-04", periods=1500)
    events = dates[10:1450:15]
    hawk = rng.random(len(events)) > 0.5
    texts = [("The Committee will raise its target. " if h else "The Committee will lower its target. ")
             + "Inflation was stable. Voting for the action were members." for h in hawk]
    pd.DataFrame({"date": events, "url": "x", "text": texts}).to_csv(tmp_path / "data" / "statements.csv", index=False)

    # Yields jump on hawkish days, so the tone surprise should explain the move.
    y2 = pd.Series(2.0 + rng.normal(0, 0.01, len(dates)).cumsum(), index=dates)
    for d, h in zip(events, hawk):
        y2.loc[d:] += 0.05 if h else -0.05
    series = {"DGS2": y2, "DGS10": y2 + 1, "DTWEXBGS": pd.Series(100.0, index=dates), "DTWEXM": pd.Series(np.nan, index=dates),
              "DFEDTAR": pd.Series(np.nan, index=dates), "DFEDTARU": pd.Series(1.0, index=dates)}
    for sid, s in series.items():
        pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"), sid: s.to_numpy()}).to_csv(
            tmp_path / "data" / "fred" / f"{sid}.csv", index=False)

    train = pd.DataFrame({"sentence": ["raise rates", "lower rates", "meeting held"] * 4,
                          "label": ["hawkish", "dovish", "neutral"] * 4, "year": 2000})
    monkeypatch.setattr(cli, "load_split", lambda split, data_dir: train)
    monkeypatch.setattr(cli, "make_provider", lambda name, model, cache: KeywordProvider(cache_dir=cache))

    cli.main(["score", "--provider", "anthropic", "--runs", "2", "--workers", "2"])
    scores = pd.read_csv(tmp_path / "data" / "scores.csv")
    assert len(scores) == len(events)
    assert scores["llm_quotes_verified"].all()
    assert {"llm_stance", "llm_net_tone", "lexicon_tone", "tfidf_tone"} <= set(scores.columns)

    cli.main(["analyze"])
    rep = json.loads((tmp_path / "results" / "events.json").read_text())
    llm = rep["explain"]["d2y"]["LLM document stance"]
    assert llm["beta"] > 1 and llm["t"] > 5
    assert (tmp_path / "figures" / "surprise_vs_2y.png").exists()
