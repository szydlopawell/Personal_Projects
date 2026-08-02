import numpy as np
import pandas as pd

from forecasting import build_features, ewma_forecast, naive_persistence_forecast, train_test_columns, walk_forward_evaluate


def make_ohlc(n=300, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.standard_normal(n) * 0.001)
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n)) * 0.001)
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n)) * 0.001)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})


def test_walk_forward_evaluate_returns_one_row_per_fold():
    ohlc = make_ohlc()
    data = build_features(ohlc, windows=(5, 10, 20), horizon=10)
    X, y = train_test_columns(data)
    naive = naive_persistence_forecast(data)
    ewma = ewma_forecast(ohlc, span=20).loc[data.index]

    results = walk_forward_evaluate(X, y, naive, ewma, n_splits=4)
    assert len(results) == 4
    assert (results[["rmse_ml", "rmse_naive", "rmse_ewma"]] >= 0).all().all()
