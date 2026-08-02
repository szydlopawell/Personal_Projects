import numpy as np
import pandas as pd

from volatility import yang_zhang_volatility

WINDOW = 10


def make_ohlc(n=30, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    open_ = close + rng.standard_normal(n) * 0.1
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n)) * 0.1
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * 0.1
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})


def test_constant_price_has_zero_volatility():
    n = 25
    flat = pd.DataFrame({"Open": [100.0] * n, "High": [100.0] * n, "Low": [100.0] * n, "Close": [100.0] * n})
    vol = yang_zhang_volatility(flat, window=WINDOW)
    assert (vol.dropna() == 0).all()


def test_leading_nans_equal_window_length():
    ohlc = make_ohlc()
    vol = yang_zhang_volatility(ohlc, window=WINDOW)
    assert vol.iloc[:WINDOW].isna().all()
    assert vol.notna().iloc[WINDOW:].all()


def test_output_length_matches_input():
    ohlc = make_ohlc(n=40)
    vol = yang_zhang_volatility(ohlc, window=WINDOW)
    assert len(vol) == len(ohlc)


def test_volatility_is_nonnegative():
    ohlc = make_ohlc(n=50, seed=1)
    vol = yang_zhang_volatility(ohlc, window=WINDOW)
    assert (vol.dropna() >= 0).all()


def test_annualization_scales_with_sqrt_trading_periods():
    ohlc = make_ohlc(n=40, seed=2)
    vol_252 = yang_zhang_volatility(ohlc, window=WINDOW, trading_periods=252)
    vol_12 = yang_zhang_volatility(ohlc, window=WINDOW, trading_periods=12)
    ratio = (vol_252 / vol_12).dropna()
    assert np.allclose(ratio.values, np.sqrt(252 / 12))
