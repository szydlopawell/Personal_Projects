"""Closed-form Black-Scholes European option price, used to validate the
Monte Carlo engine (the European MC price should converge to this)."""

import numpy as np
from scipy.stats import norm


def black_scholes_price(S0: float, K: float, r: float, T: float, sigma: float, option_type: str = "call") -> float:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
