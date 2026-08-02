"""Simple, non-ML volatility forecasts used to benchmark the learned model.

Both are legitimate forecasts in their own right -- and hard for a model to
beat, which is precisely why they're the right bar to clear rather than a
trivial strawman.
"""

import numpy as np


def naive_persistence_forecast(features):
    """Forecast next-period vol as simply today's 20-day realized vol."""
    return features["yz_vol_20"]


def ewma_forecast(ohlc, span=20, trading_periods=252):
    """Exponentially-weighted realized vol, evaluated as of each row."""
    returns = np.log(ohlc["Close"] / ohlc["Close"].shift(1))
    return returns.ewm(span=span).std() * np.sqrt(trading_periods)
