"""Feature and target construction for volatility forecasting.

The target is the realized close-to-close volatility over the *next*
`horizon` days -- strictly future data relative to each row's timestamp.
Every feature, by contrast, is built only from Yang-Zhang windows and
momentum computed up to and including that row, so there is no leakage of
future information into the inputs.
"""

import numpy as np
import pandas as pd

from volatility import yang_zhang_volatility


def build_features(ohlc: pd.DataFrame, windows=(5, 10, 20, 60), horizon: int = 20,
                    trading_periods: int = 252) -> pd.DataFrame:
    returns = np.log(ohlc["Close"] / ohlc["Close"].shift(1))

    data = pd.DataFrame(index=ohlc.index)
    for w in windows:
        data[f"yz_vol_{w}"] = yang_zhang_volatility(ohlc, window=w, trading_periods=trading_periods)

    data["momentum_5"] = ohlc["Close"].pct_change(5)
    data["vol_of_vol_20"] = data["yz_vol_20"] - data["yz_vol_20"].shift(5)

    # Forward realized vol: the trailing `horizon`-window sum of squared
    # returns, evaluated `horizon` steps ahead and shifted back to align with
    # today's row, so target[t] = realized vol over (t, t+horizon].
    forward_sq_return_sum = (returns**2).rolling(horizon).sum().shift(-horizon)
    data["target"] = np.sqrt(forward_sq_return_sum / horizon * trading_periods)

    return data.dropna()


def train_test_columns(data: pd.DataFrame):
    feature_cols = [c for c in data.columns if c != "target"]
    return data[feature_cols], data["target"]
