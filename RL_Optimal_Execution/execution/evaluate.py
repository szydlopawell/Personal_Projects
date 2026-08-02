"""Monte Carlo evaluation of a fixed trade schedule's realized implementation
shortfall, using the same price/impact dynamics as the training environment."""

import numpy as np

from .environment import execution_price, shortfall_cost, update_price


def simulate_shortfall(shares_per_step, S0, sigma, tau, eta, gamma, rng):
    price = S0
    total_cost = 0.0
    for n in shares_per_step:
        exec_p = execution_price(price, n, tau, eta)
        total_cost += shortfall_cost(S0, n, exec_p)
        z = rng.standard_normal()
        price = update_price(price, n, sigma, tau, gamma, z)
    return total_cost


def monte_carlo_shortfall(shares_per_step, S0, sigma, tau, eta, gamma, n_sims=5000, seed=1):
    rng = np.random.default_rng(seed)
    costs = np.array([
        simulate_shortfall(shares_per_step, S0, sigma, tau, eta, gamma, rng)
        for _ in range(n_sims)
    ])
    return costs
