"""Tabular Q-learning for the optimal-execution problem.

State is (time step, inventory remaining in lots) -- price is deliberately
excluded. Under Almgren-Chriss's linear-impact, martingale-price assumptions
the optimal remaining schedule doesn't depend on the price path realized so
far (only sunk cost, not the forward-looking decision), so this smaller
state space is sufficient and trains far faster than including price.
"""

import numpy as np


def train_q_learning(env, n_episodes=20_000, alpha=0.1, epsilon_start=1.0,
                      epsilon_end=0.05, epsilon_decay_episodes=15_000, seed=0):
    rng = np.random.default_rng(seed)
    N, n_lots = env.N, env.n_lots
    Q = {(k, inv): np.zeros(inv + 1) for k in range(N) for inv in range(n_lots + 1)}

    for ep in range(n_episodes):
        epsilon = np.interp(ep, [0, epsilon_decay_episodes], [epsilon_start, epsilon_end])
        state = env.reset()
        done = False
        while not done:
            k, inv = state
            if inv == 0:
                action = 0
            elif rng.random() < epsilon:
                action = rng.integers(0, inv + 1)
            else:
                action = int(np.argmax(Q[(k, inv)]))

            next_state, reward, done, _ = env.step(action)
            best_next = 0.0 if done else np.max(Q[next_state])
            Q[(k, inv)][action] += alpha * (reward + best_next - Q[(k, inv)][action])
            state = next_state

    return Q


def extract_policy_trajectory(Q, N, n_lots):
    inv = n_lots
    trajectory = [inv]
    for k in range(N):
        action = inv if k == N - 1 else (int(np.argmax(Q[(k, inv)])) if inv > 0 else 0)
        inv -= action
        trajectory.append(inv)
    return np.array(trajectory)
