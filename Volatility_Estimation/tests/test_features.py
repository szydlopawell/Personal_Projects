import numpy as np
import pandas as pd

from forecasting import build_features

WINDOWS = (5, 10, 20)
HORIZON = 10


def make_ohlc(n=150, seed=0, shock_at=None, shock_size=0.0):
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n) * 0.01
    if shock_at is not None:
        returns[shock_at] += shock_size
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.standard_normal(n) * 0.001)
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n)) * 0.001)
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n)) * 0.001)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})


def test_features_do_not_leak_a_future_shock():
    n = 150
    shock_at = n - 1
    base = make_ohlc(n=n, seed=1)
    shocked = make_ohlc(n=n, seed=1, shock_at=shock_at, shock_size=0.5)

    feat_base = build_features(base, windows=WINDOWS, horizon=HORIZON)
    feat_shocked = build_features(shocked, windows=WINDOWS, horizon=HORIZON)

    feature_cols = [c for c in feat_base.columns if c != "target"]
    common_idx = feat_base.index.intersection(feat_shocked.index)
    safe_idx = [i for i in common_idx if i < shock_at - max(WINDOWS)]

    assert len(safe_idx) > 10
    pd.testing.assert_frame_equal(
        feat_base.loc[safe_idx, feature_cols], feat_shocked.loc[safe_idx, feature_cols]
    )


def test_target_reflects_a_future_shock():
    n = 150
    shock_at = n - 1
    base = make_ohlc(n=n, seed=1)
    shocked = make_ohlc(n=n, seed=1, shock_at=shock_at, shock_size=0.5)

    feat_base = build_features(base, windows=WINDOWS, horizon=HORIZON)
    feat_shocked = build_features(shocked, windows=WINDOWS, horizon=HORIZON)

    # The last row whose forward-looking window [t+1, t+HORIZON] both
    # reaches the shock and still has a non-NaN target (t+HORIZON <= n-1).
    row_near_shock = shock_at - HORIZON
    assert row_near_shock in feat_base.index
    assert row_near_shock in feat_shocked.index
    assert feat_shocked.loc[row_near_shock, "target"] > feat_base.loc[row_near_shock, "target"] * 2


def test_output_has_no_nans():
    ohlc = make_ohlc(n=200, seed=2)
    data = build_features(ohlc, windows=WINDOWS, horizon=HORIZON)
    assert not data.isna().any().any()


def test_target_column_present():
    ohlc = make_ohlc(n=200, seed=3)
    data = build_features(ohlc, windows=WINDOWS, horizon=HORIZON)
    assert "target" in data.columns
    assert (data["target"] >= 0).all()
