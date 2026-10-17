import numpy as np
import pandas as pd
import pytest

from fomc_signal.events import add_surprises, build_market_frame, drift_trade, event_frame, explain_moves, out_of_sample


def market(n=40):
    idx = pd.bdate_range("2024-01-01", periods=n)
    fred = pd.DataFrame(index=idx)
    fred["DGS2"] = 4.0 + 0.01 * np.arange(n)
    fred["DGS10"] = 4.5
    fred["DTWEXBGS"] = 100 * np.exp(0.001 * np.arange(n))
    fred["DTWEXM"] = np.nan
    fred["DFEDTAR"] = np.nan
    fred["DFEDTARU"] = 5.5
    fred.loc[idx[20]:, "DFEDTARU"] = 5.75
    return build_market_frame(fred)


def test_event_frame_alignment():
    mkt = market()
    ev = event_frame(pd.Series([mkt.index[20]]), mkt, drift_days=5)
    row = ev.iloc[0]
    assert row["d2y"] == pytest.approx(1.0)  # +1bp per day
    assert row["dtarget"] == pytest.approx(25.0)
    assert row["drift_2y"] == pytest.approx(5.0)
    assert row["dusd"] == pytest.approx(0.1)


def test_weekend_statement_rolls_to_next_trading_day():
    mkt = market()
    saturday = mkt.index[4] + pd.Timedelta(days=1)  # 2024-01-06
    ev = event_frame(pd.Series([saturday]), mkt)
    assert ev.iloc[0]["trade_date"] == mkt.index[5]


def synthetic_events(n=200, beta=4.0, seed=0):
    rng = np.random.default_rng(seed)
    tone = np.cumsum(rng.normal(0, 1, n))
    dtarget = rng.choice([0, 0, 0, 25, -25], n)
    surprise = np.r_[np.nan, np.diff(tone)]
    s_std = np.nanstd(surprise, ddof=1)
    d2y = 0.2 * dtarget + beta * surprise / s_std + rng.normal(0, 3, n)
    drift = 2.0 * np.sign(surprise) + rng.normal(0, 4, n)
    return pd.DataFrame({"date": pd.date_range("2000-01-31", periods=n, freq="45D"), "tone": tone,
                         "d2y": d2y, "dtarget": dtarget, "drift_2y": drift})


def test_regression_recovers_tone_effect():
    ev = add_surprises(synthetic_events(), ["tone"])
    res = explain_moves(ev, "tone")
    assert res["beta"] == pytest.approx(4.0, abs=0.6)
    assert res["t"] > 8
    assert res["r2"] > res["r2_policy_only"]


def test_out_of_sample_beats_policy_only_when_tone_matters():
    ev = add_surprises(synthetic_events(), ["tone"])
    res = out_of_sample(ev, "tone", min_train=60)
    assert res["oos_r2_vs_policy_only"] > 0.3
    assert res["hit_rate"] > 0.7


def test_drift_trade_sign_convention():
    ev = add_surprises(synthetic_events(), ["tone"])
    res = drift_trade(ev, "tone")
    assert res["mean_bp"] == pytest.approx(2.0, abs=0.8)
