"""Event study: do tone surprises move rates and the dollar?

For each statement date d:

* ``d2y`` is the close-to-close change in the 2-year Treasury yield from the
  previous trading day to d, in basis points (statements come out at 2pm ET,
  so the day-d close reflects them).
* ``dusd`` is the log change in the trade-weighted dollar over the same
  window, in percent. The broad index (DTWEXBGS) starts in 2006; earlier
  changes come from the major-currencies index (DTWEXM).
* ``dtarget`` is the change in the fed funds target (upper bound since Dec
  2008), the part of the decision the statement tone has to add to.
* ``drift_2y`` is the change from the close on d to the close five trading
  days later, which a desk could actually trade after reading the statement.

A tone *surprise* is the change in a tone measure from the previous
statement, since markets react to news rather than levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

FRED_SERIES = ["DGS2", "DGS10", "DTWEXBGS", "DTWEXM", "DFEDTAR", "DFEDTARU"]


def build_market_frame(fred: pd.DataFrame) -> pd.DataFrame:
    """Daily frame with 2y/10y yields (%), dollar log-change series and policy target."""
    df = pd.DataFrame(index=fred.index)
    df["y2"] = fred["DGS2"]
    df["y10"] = fred["DGS10"]
    broad = np.log(fred["DTWEXBGS"]).diff()
    major = np.log(fred["DTWEXM"]).diff()
    df["usd_ret"] = broad.where(broad.notna() & fred["DTWEXBGS"].notna(), major) * 100
    df["target"] = fred["DFEDTARU"].combine_first(fred["DFEDTAR"])
    return df


def event_frame(dates: pd.Series, market: pd.DataFrame, drift_days: int = 5) -> pd.DataFrame:
    """Market moves around each statement date (dates not trading days roll forward)."""
    trading = market.dropna(subset=["y2"]).index
    target = market["target"].ffill()
    rows = []
    for d in pd.to_datetime(dates):
        pos = trading.searchsorted(d)
        if pos == 0 or pos + drift_days >= len(trading):
            continue
        t, prev, fwd = trading[pos], trading[pos - 1], trading[pos + drift_days]
        y2 = market["y2"]
        usd = market["usd_ret"].loc[t] if not pd.isna(market["usd_ret"].loc[t]) else np.nan
        rows.append({
            "date": d,
            "trade_date": t,
            "d2y": (y2.loc[t] - y2.loc[prev]) * 100,
            "d10y": (market["y10"].loc[t] - market["y10"].loc[prev]) * 100,
            "dusd": usd,
            "dtarget": (target.loc[t] - target.loc[prev]) * 100,
            "drift_2y": (y2.loc[fwd] - y2.loc[t]) * 100,
        })
    return pd.DataFrame(rows)


def add_surprises(df: pd.DataFrame, measures: list[str]) -> pd.DataFrame:
    """Change of each tone measure versus the previous statement, standardised to unit variance."""
    out = df.sort_values("date").copy()
    for m in measures:
        d = out[m].diff()
        out[f"s_{m}"] = d / d.std()
    return out


def hac_ols(y: pd.Series, X: pd.DataFrame, lags: int = 2):
    data = pd.concat([y, X], axis=1).dropna()
    model = sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:]))
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def explain_moves(ev: pd.DataFrame, measure: str, outcome: str = "d2y") -> dict:
    """Regress the market move on the tone surprise, controlling for the rate decision."""
    base = hac_ols(ev[outcome], ev[["dtarget"]])
    full = hac_ols(ev[outcome], ev[["dtarget", f"s_{measure}"]])
    return {
        "beta": float(full.params[f"s_{measure}"]),  # bp (or %) per 1 s.d. tone surprise
        "t": float(full.tvalues[f"s_{measure}"]),
        "r2": float(full.rsquared),
        "r2_policy_only": float(base.rsquared),
        "n": int(full.nobs),
    }


def out_of_sample(ev: pd.DataFrame, measure: str, outcome: str = "d2y", min_train: int = 60) -> dict:
    """Expanding-window forecasts: does adding tone beat the decision-only model out of sample?"""
    data = ev[["date", outcome, "dtarget", f"s_{measure}"]].dropna().sort_values("date").reset_index(drop=True)
    preds_full, preds_base, actual = [], [], []
    for i in range(min_train, len(data)):
        train = data.iloc[:i]
        x_full = sm.add_constant(train[["dtarget", f"s_{measure}"]])
        x_base = sm.add_constant(train[["dtarget"]])
        b_full = sm.OLS(train[outcome], x_full).fit().params
        b_base = sm.OLS(train[outcome], x_base).fit().params
        row = data.iloc[i]
        preds_full.append(b_full["const"] + b_full["dtarget"] * row["dtarget"] + b_full[f"s_{measure}"] * row[f"s_{measure}"])
        preds_base.append(b_base["const"] + b_base["dtarget"] * row["dtarget"])
        actual.append(row[outcome])
    a, pf, pb = map(np.asarray, (actual, preds_full, preds_base))
    if len(a) == 0:
        return {"n": 0}
    sse_full, sse_base = np.sum((a - pf) ** 2), np.sum((a - pb) ** 2)
    nz = a != 0
    return {
        "n": int(len(a)),
        "oos_r2_vs_policy_only": float(1 - sse_full / sse_base),
        "hit_rate": float(np.mean(np.sign(pf[nz]) == np.sign(a[nz]))),
    }


def drift_trade(ev: pd.DataFrame, measure: str, outcome: str = "drift_2y") -> dict:
    """Trade the post-statement drift: short 2y bonds after a hawkish surprise, long after a dovish one.

    P&L per event is ``sign(surprise) * change in yield`` in basis points of
    yield, i.e. positive when yields keep moving in the direction of the tone.
    """
    data = ev[[outcome, f"s_{measure}"]].dropna()
    data = data[data[f"s_{measure}"] != 0]
    pnl = np.sign(data[f"s_{measure}"]) * data[outcome]
    if len(pnl) < 3:
        return {"n": int(len(pnl))}
    return {
        "n": int(len(pnl)),
        "mean_bp": float(pnl.mean()),
        "t": float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl))),
        "hit_rate": float((pnl > 0).mean()),
    }
