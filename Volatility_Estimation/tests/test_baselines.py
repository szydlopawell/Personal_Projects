import numpy as np
import pandas as pd

from forecasting.baselines import ewma_forecast, naive_persistence_forecast


def test_naive_persistence_equals_yz_vol_20_column():
    df = pd.DataFrame({"yz_vol_20": [0.1, 0.2, 0.15], "other_col": [1, 2, 3]})
    result = naive_persistence_forecast(df)
    pd.testing.assert_series_equal(result, df["yz_vol_20"])


def test_ewma_forecast_is_nonnegative():
    n = 60
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.standard_normal(n))
    ohlc = pd.DataFrame({"Close": close})
    vol = ewma_forecast(ohlc, span=20)
    assert (vol.dropna() >= 0).all()


def test_ewma_forecast_zero_for_constant_price():
    n = 30
    ohlc = pd.DataFrame({"Close": [100.0] * n})
    vol = ewma_forecast(ohlc, span=10)
    assert (vol.dropna() == 0).all()
