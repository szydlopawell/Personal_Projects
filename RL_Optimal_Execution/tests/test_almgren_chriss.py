import numpy as np

from execution.almgren_chriss import ac_trajectory, twap_trajectory

X, T, N, SIGMA, ETA = 10_000, 1.0, 10, 0.3, 2.5e-6


def test_ac_trajectory_starts_at_full_inventory():
    traj = ac_trajectory(X, T, N, SIGMA, ETA, lam=1e-4)
    assert traj[0] == X


def test_ac_trajectory_ends_at_zero():
    traj = ac_trajectory(X, T, N, SIGMA, ETA, lam=1e-4)
    assert abs(traj[-1]) < 1e-6


def test_ac_trajectory_is_monotonically_decreasing():
    traj = ac_trajectory(X, T, N, SIGMA, ETA, lam=1e-4)
    assert (np.diff(traj) <= 0).all()


def test_zero_risk_aversion_reduces_to_twap():
    ac_traj = ac_trajectory(X, T, N, SIGMA, ETA, lam=0.0)
    twap_traj = twap_trajectory(X, N)
    assert np.allclose(ac_traj, twap_traj)


def test_higher_risk_aversion_front_loads_more_than_lower():
    low_lam = ac_trajectory(X, T, N, SIGMA, ETA, lam=1e-5)
    high_lam = ac_trajectory(X, T, N, SIGMA, ETA, lam=1e-3)
    # More risk-averse (higher lam) should hold less inventory at every
    # intermediate step -- it liquidates faster to cut price-risk exposure.
    assert (high_lam[1:-1] <= low_lam[1:-1]).all()


def test_twap_trajectory_is_linear():
    traj = twap_trajectory(X, N)
    assert traj[0] == X
    assert traj[-1] == 0
    assert np.allclose(np.diff(traj), -X / N)
