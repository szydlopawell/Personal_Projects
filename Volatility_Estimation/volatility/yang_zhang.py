"""Yang-Zhang (2000) volatility estimator.

Combines overnight (close-to-open) variance, open-to-close variance, and the
drift-independent Rogers-Satchell term into a single minimum-variance
estimator that is robust to both opening jumps and intraday drift.

Reference: Yang, D. and Zhang, Q. (2000), "Drift-Independent Volatility
Estimation Based on High, Low, Open, and Close Prices", Journal of Business.
"""

import numpy as np
import pandas as pd


def yang_zhang_volatility(
    ohlc: pd.DataFrame,
    window: int = 20,
    trading_periods: int = 252,
) -> pd.Series:
    """Rolling annualized Yang-Zhang volatility.

    Parameters
    ----------
    ohlc : DataFrame with columns "Open", "High", "Low", "Close", indexed by date.
    window : rolling window size in trading days.
    trading_periods : periods per year used to annualize (252 for daily equities).

    Returns
    -------
    Series of annualized volatility, aligned to ohlc.index.
    """
    o = ohlc["Open"]
    h = ohlc["High"]
    l = ohlc["Low"]
    c = ohlc["Close"]
    prev_c = c.shift(1)

    overnight = np.log(o / prev_c)
    open_to_close = np.log(c / o)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)

    # Rogers-Satchell per-day term: unbiased on its own, so it is averaged
    # with ddof=0 rather than the (n-1) sample-variance correction used below.
    rs = log_ho * (log_ho - open_to_close) + log_lo * (log_lo - open_to_close)

    var_overnight = overnight.rolling(window).var(ddof=1)
    var_open_close = open_to_close.rolling(window).var(ddof=1)
    var_rs = rs.rolling(window).mean()

    # k minimizes the variance of the combined estimator (Yang & Zhang 2000, eq. 18).
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    variance = var_overnight + k * var_open_close + (1 - k) * var_rs
    return np.sqrt(variance * trading_periods)
