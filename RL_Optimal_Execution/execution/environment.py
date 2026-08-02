"""Optimal-execution environment under Almgren-Chriss price impact.

An agent liquidates X shares over N steps of length tau = T/N. Selling n
shares in a step both (a) permanently shifts the price via gamma*n, so all
future fills are affected, and (b) executes at a worse price via temporary
impact eta*(n/tau) that only affects that step's own fill.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def execution_price(price, n, tau, eta):
    return price - eta * (n / tau)


def shortfall_cost(S0, n, exec_price):
    return S0 * n - n * exec_price


def update_price(price, n, sigma, tau, gamma, z):
    return price + sigma * np.sqrt(tau) * z - gamma * n


@dataclass
class ExecutionEnv:
    X: float = 10_000
    T: float = 1.0
    N: int = 10
    S0: float = 50.0
    sigma: float = 0.3
    eta: float = 2.5e-6
    gamma: float = 2.5e-7
    lam: float = 1e-4
    n_lots: int = 20
    seed: Optional[int] = None

    def __post_init__(self):
        self.tau = self.T / self.N
        self.lot_size = self.X / self.n_lots
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self):
        self.k = 0
        self.inv_lots = self.n_lots
        self.price = self.S0
        return (self.k, self.inv_lots)

    def step(self, sell_lots):
        sell_lots = int(np.clip(sell_lots, 0, self.inv_lots))
        if self.k == self.N - 1:
            sell_lots = self.inv_lots  # force full liquidation by the horizon

        n = sell_lots * self.lot_size
        inv_before_shares = self.inv_lots * self.lot_size

        exec_p = execution_price(self.price, n, self.tau, self.eta)
        cost = shortfall_cost(self.S0, n, exec_p)
        # Holding x shares over a period of length tau contributes sigma^2 *
        # x^2 * tau to Var[total cost]; penalizing it directly is what makes
        # the agent front-load execution instead of converging to plain TWAP.
        risk_penalty = self.lam * self.sigma**2 * inv_before_shares**2 * self.tau

        z = self.rng.standard_normal()
        self.price = update_price(self.price, n, self.sigma, self.tau, self.gamma, z)
        self.inv_lots -= sell_lots
        self.k += 1

        reward = -(cost + risk_penalty)
        done = self.k == self.N
        return (self.k, self.inv_lots), reward, done, {"cost": cost}
