"""Closed-form Almgren-Chriss (2000) optimal execution trajectory, and the
naive TWAP (uniform) baseline it reduces to when risk aversion is 0."""

import numpy as np


def ac_trajectory(X, T, N, sigma, eta, lam):
    kappa = np.sqrt(lam * sigma**2 / eta)
    t = np.linspace(0, T, N + 1)
    if kappa * T < 1e-8:
        return X * (1 - t / T)
    return X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)


def twap_trajectory(X, N):
    return X * (1 - np.arange(N + 1) / N)
