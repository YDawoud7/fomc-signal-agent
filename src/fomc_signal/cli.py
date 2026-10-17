"""Command line entry point.

    fomc-signal fetch                      # statements + market data -> data/
    fomc-signal validate --provider anthropic
    fomc-signal score --provider anthropic --runs 3
    fomc-signal analyze
    fomc-signal all --provider anthropic   # everything above

Providers: anthropic (ANTHROPIC_API_KEY), openai (OPENAI_API_KEY) or
claude-code (uses the local `claude` CLI, no API key needed).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import events, lexicon
from .baselines import TfidfClassifier
from .fed import fetch_statements
from .labels import evaluate, load_split
from .llm import make_provider
from .scoring import label_sentences, score_statements
from .text import policy_sentences

DATA, RESULTS, FIG = Path("data"), Path("results"), Path("figures")


# ---------------------------------------------------------------- fetch
def fetch_fred(series: list[str], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for sid in series:
        path = out_dir / f"{sid}.csv"
        if not path.exists():
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            req = urllib.request.Request(url, headers={"User-Agent": "fomc-signal/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                path.write_bytes(resp.read())
            time.sleep(0.5)
        df = pd.read_csv(path, na_values=["."])
        df = df.set_index(pd.to_datetime(df.iloc[:, 0])).iloc[:, 1:]
        df.columns = [sid]
        frames.append(df)
    return pd.concat(frames, axis=1).sort_index()


def cmd_fetch(args) -> None:
    print("Downloading FOMC statements ...")
    st = fetch_statements(start_year=args.start_year, cache_dir=DATA / "raw")
    st.to_csv(DATA / "statements.csv", index=False)
    print(f"  {len(st)} statements, {st['date'].min():%Y-%m-%d} to {st['date'].max():%Y-%m-%d}")
    print("Downloading market data from FRED ...")
    fetch_fred(events.FRED_SERIES, DATA / "fred")
    print("Done.")


def load_statements() -> pd.DataFrame:
    return pd.read_csv(DATA / "statements.csv", parse_dates=["date"])


# ---------------------------------------------------------------- validate
def cmd_validate(args) -> None:
    RESULTS.mkdir(exist_ok=True)
    train, test = load_split("train", DATA / "tdw"), load_split("test", DATA / "tdw")
    results = {}
    results["Dictionary"] = evaluate(test.label, [lexicon.classify(s) for s in test.sentence])
    tfidf = TfidfClassifier().fit(train.sentence, train.label)
    results["TF-IDF logistic regression"] = evaluate(test.label, tfidf.predict(test.sentence))
    if args.roberta:
        from .baselines import RobertaClassifier

        results["FOMC-RoBERTa (fine-tuned)"] = evaluate(test.label, RobertaClassifier().predict(test.sentence))

    provider = make_provider(args.provider, args.model, DATA / "llm_cache")
    batches = [list(test.sentence[i : i + 25]) for i in range(0, len(test), 25)]
    with ThreadPoolExecutor(args.workers) as pool:
        preds = [lab for out in pool.map(lambda b: label_sentences(provider, b), batches) for lab in out]
    results[f"LLM ({provider.model})"] = evaluate(test.label, preds)
    results["_usage"] = provider.usage.__dict__

    (RESULTS / "validation.json").write_text(json.dumps(results, indent=2))
    lines = ["| Classifier | Accuracy | Macro F1 |", "| --- | ---: | ---: |"]
    for k, v in results.items():
        if not k.startswith("_"):
            lines.append(f"| {k} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
    (RESULTS / "validation.md").write_text("\n".join(lines) + f"\n\nTest set: {len(test)} human-labelled sentences.\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- score
def _net(labels: list[str]) -> float:
    return (labels.count("hawkish") - labels.count("dovish")) / max(len(labels), 1)


def baseline_tones(st: pd.DataFrame, roberta: bool = False) -> pd.DataFrame:
    """Net tone of each statement under the non-LLM classifiers."""
    train = load_split("train", DATA / "tdw")
    tfidf = TfidfClassifier().fit(train.sentence, train.label)
    sents = [policy_sentences(t) for t in st["text"]]
    out = pd.DataFrame({"date": st["date"].to_numpy()})
    out["lexicon_tone"] = [lexicon.document_score(s) for s in sents]
    out["tfidf_tone"] = [_net(tfidf.predict(s)) for s in sents]
    if roberta:
        from .baselines import RobertaClassifier

        rob = RobertaClassifier()
        out["roberta_tone"] = [_net(rob.predict(s)) for s in sents]
    return out


def cmd_score(args) -> None:
    st = load_statements().sort_values("date").reset_index(drop=True)
    provider = make_provider(args.provider, args.model, DATA / "llm_cache")
    n = max(1, args.workers)
    chunks = [st.iloc[i::n] for i in range(n)]
    with ThreadPoolExecutor(n) as pool:
        parts = list(pool.map(lambda c: score_statements(c, provider, runs=args.runs, progress=False), chunks))
    scores = pd.concat(parts).sort_values("date")
    scores = scores.merge(baseline_tones(st, args.roberta), on="date", how="left")
    scores.to_csv(DATA / "scores.csv", index=False)
    u = provider.usage
    print(f"Scored {len(scores)} statements: {u.calls} LLM calls ({u.cached} cached), "
          f"{u.input_tokens + u.output_tokens:,} tokens, est. ${u.cost_usd:.2f}")


# ---------------------------------------------------------------- analyze
MEASURES = {
    "llm_stance": "LLM document stance",
    "llm_net_tone": "LLM sentence net tone",
    "roberta_tone": "FOMC-RoBERTa net tone",
    "tfidf_tone": "TF-IDF net tone",
    "lexicon_tone": "Dictionary net tone",
}


def cmd_analyze(args) -> None:
    RESULTS.mkdir(exist_ok=True)
    scores = pd.read_csv(DATA / "scores.csv", parse_dates=["date"])
    fred = fetch_fred(events.FRED_SERIES, DATA / "fred")
    ev = events.event_frame(scores["date"], events.build_market_frame(fred))
    df = ev.merge(scores, on="date")
    # Skip measures that are missing or never change (nothing to regress on).
    measures = [m for m in MEASURES if m in df.columns and df[m].diff().std() > 0]
    df = events.add_surprises(df, measures)
    df.to_csv(RESULTS / "event_panel.csv", index=False)

    report = {"n_events": int(len(df)), "start": str(df["date"].min().date()), "end": str(df["date"].max().date())}
    tables = {}
    for outcome, label in [("d2y", "2y yield (bp)"), ("dusd", "Dollar index (%)"), ("d10y", "10y yield (bp)")]:
        rows = {}
        for m in measures:
            r = events.explain_moves(df, m, outcome)
            o = events.out_of_sample(df, m, outcome)
            rows[MEASURES[m]] = {**r, **{f"oos_{k}": v for k, v in o.items()}}
        tables[outcome] = rows
    report["explain"] = tables
    report["drift_2y"] = {MEASURES[m]: events.drift_trade(df, m) for m in measures}
    split = df["date"] < "2013-01-01"
    report["subsample_2y"] = {
        period: {MEASURES[m]: events.explain_moves(df[mask], m, "d2y") for m in measures}
        for period, mask in [("2000-2012", split), ("2013-2026", ~split)]
    }
    if "llm_stance_std" in df:
        report["llm_consistency"] = {
            "mean_std_across_runs": float(df["llm_stance_std"].mean()),
            "share_quotes_verified": float(df["llm_quotes_verified"].mean()),
        }
    report["tone_correlation"] = df[measures].corr().round(3).to_dict()
    (RESULTS / "events.json").write_text(json.dumps(report, indent=2, default=float))
    md = _markdown(report, measures)
    (RESULTS / "events.md").write_text(md)
    _plots(df, fred)
    print(md)


def _markdown(rep: dict, measures: list[str]) -> str:
    out = [f"Events: {rep['n_events']} statements, {rep['start']} to {rep['end']}.", ""]
    names = {"d2y": "2-year Treasury yield", "dusd": "Trade-weighted dollar", "d10y": "10-year Treasury yield"}
    for outcome, rows in rep["explain"].items():
        unit = "%" if outcome == "dusd" else "bp"
        out += [f"### {names[outcome]} on statement day", "",
                f"| Tone surprise measure | Effect per 1 s.d. ({unit}) | t-stat | R² | R² decision only | OOS R² gain | OOS hit rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, r in rows.items():
            out.append(f"| {name} | {r['beta']:.2f} | {r['t']:.2f} | {r['r2']:.3f} | {r['r2_policy_only']:.3f} | "
                       f"{r.get('oos_oos_r2_vs_policy_only', float('nan')):.3f} | {r.get('oos_hit_rate', float('nan')):.2f} |")
        out.append("")
    out += ["### Post-statement drift trade (2y, next 5 trading days)", "",
            "| Measure | Events | Mean P&L (bp) | t-stat | Hit rate |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, r in rep["drift_2y"].items():
        if r.get("n", 0) >= 3:
            out.append(f"| {name} | {r['n']} | {r['mean_bp']:.2f} | {r['t']:.2f} | {r['hit_rate']:.2f} |")
    out += ["", "### Sub-samples (2y yield)", "", "| Measure | Period | Effect (bp) | t-stat | R² |", "| --- | --- | ---: | ---: | ---: |"]
    for period, rows in rep["subsample_2y"].items():
        for name, r in rows.items():
            out.append(f"| {name} | {period} | {r['beta']:.2f} | {r['t']:.2f} | {r['r2']:.3f} |")
    if "llm_consistency" in rep:
        c = rep["llm_consistency"]
        out += ["", f"LLM consistency: average std of the stance score across repeated runs {c['mean_std_across_runs']:.3f}; "
                    f"supporting quotes verified verbatim in {c['share_quotes_verified']:.0%} of statements."]
    return "\n".join(out) + "\n"


def _plots(df: pd.DataFrame, fred: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax1.plot(df["date"], df["llm_stance"], color="#1f3b73", lw=1.2, label="LLM stance (left)")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_ylabel("Stance (-1 dovish, +1 hawkish)")
    ax2 = ax1.twinx()
    tgt = fred["DFEDTARU"].combine_first(fred["DFEDTAR"]).loc[df["date"].min():]
    ax2.plot(tgt.index, tgt, color="#e76f51", lw=1.2, label="Fed funds target (right)")
    ax2.set_ylabel("%")
    ax1.set_title("LLM-scored stance of FOMC statements vs the policy rate")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "stance_vs_policy_rate.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    hold = df["dtarget"] == 0
    ax.scatter(df.loc[hold, "s_llm_stance"], df.loc[hold, "d2y"], s=14, color="#1f3b73", label="No rate change")
    ax.scatter(df.loc[~hold, "s_llm_stance"], df.loc[~hold, "d2y"], s=14, color="#e76f51", label="Rate change")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("LLM stance surprise (s.d.)")
    ax.set_ylabel("2y yield change on the day (bp)")
    ax.set_title("Tone surprises and the 2-year yield")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "surprise_vs_2y.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="fomc-signal", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def llm_args(sp):
        sp.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "claude-code"])
        sp.add_argument("--model", default=None, help="override the provider's default model")
        sp.add_argument("--workers", type=int, default=4)
        sp.add_argument("--roberta", action="store_true", help="also run the FOMC-RoBERTa baseline")

    f = sub.add_parser("fetch")
    f.add_argument("--start-year", type=int, default=2000)
    v = sub.add_parser("validate")
    llm_args(v)
    s = sub.add_parser("score")
    llm_args(s)
    s.add_argument("--runs", type=int, default=3, help="repeated document scorings to measure consistency")
    sub.add_parser("analyze")
    a = sub.add_parser("all")
    llm_args(a)
    a.add_argument("--runs", type=int, default=3)
    a.add_argument("--start-year", type=int, default=2000)

    args = p.parse_args(argv)
    DATA.mkdir(exist_ok=True)
    if args.cmd == "fetch":
        cmd_fetch(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "score":
        cmd_score(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    else:
        cmd_fetch(args)
        cmd_validate(args)
        cmd_score(args)
        cmd_analyze(args)


if __name__ == "__main__":
    main()
