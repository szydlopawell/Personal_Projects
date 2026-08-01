"""Monte Carlo option pricers under geometric Brownian motion.

- european_option_mc: plain terminal-payoff Monte Carlo, used to sanity-check
  the path simulator against the closed-form Black-Scholes price.
- american_option_lsm: Longstaff-Schwartz (2001) least-squares Monte Carlo,
  which estimates the early-exercise decision at each time step by
  regressing realized continuation payoffs onto the current price.

Both support antithetic variates: pairing each standard normal draw Z with
its mirror -Z. Since payoffs are then averaged over each pair, and the pair's
errors are negatively correlated, the variance of the estimator drops
without needing more paths.
"""

import numpy as np


def _simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic, seed):
    dt = T / n_steps
    rng = np.random.default_rng(seed)

    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_steps))

    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_paths)
    return np.hstack([np.full((S.shape[0], 1), S0), S])  # prepend S0 at t=0


def _payoff_fn(K, option_type):
    if option_type == "put":
        return lambda S: np.maximum(K - S, 0.0)
    elif option_type == "call":
        return lambda S: np.maximum(S - K, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def european_option_mc(S0, K, r, T, sigma, option_type="put", n_steps=50, n_paths=20000, seed=None, antithetic=True):
    S = _simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic, seed)
    payoff = _payoff_fn(K, option_type)

    discounted = np.exp(-r * T) * payoff(S[:, -1])
    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return price, stderr


def american_option_lsm(S0, K, r, T, sigma, option_type="put", n_steps=50, n_paths=20000, seed=None, antithetic=True, basis_degree=2):
    S = _simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic, seed)
    payoff = _payoff_fn(K, option_type)
    discount = np.exp(-r * (T / n_steps))

    cashflow = payoff(S[:, -1])  # value at maturity (t = n_steps)

    for t in range(n_steps - 1, 0, -1):
        cashflow = cashflow * discount  # roll continuation value back one step, to time t
        exercise_value = payoff(S[:, t])
        itm = exercise_value > 0
        if not np.any(itm):
            continue

        # Regress the (discounted) realized continuation value on moneyness to
        # estimate E[continuation | S_t] for in-the-money paths only, per
        # Longstaff-Schwartz -- exercising when out-of-the-money is never optimal
        # so restricting the regression to itm paths avoids fitting noise there.
        x = S[itm, t] / S0
        basis = np.vander(x, basis_degree + 1, increasing=True)
        coeffs, *_ = np.linalg.lstsq(basis, cashflow[itm], rcond=None)
        continuation_est = basis @ coeffs

        exercise_now = exercise_value[itm] > continuation_est
        idx = np.where(itm)[0][exercise_now]
        cashflow[idx] = exercise_value[itm][exercise_now]

    price_paths = cashflow * discount  # discount from t=1 to t=0
    price = price_paths.mean()
    stderr = price_paths.std(ddof=1) / np.sqrt(len(price_paths))
    return price, stderr
