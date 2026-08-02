import numpy as np
import matplotlib.pyplot as plt

from execution import (
    ExecutionEnv,
    ac_trajectory,
    extract_policy_trajectory,
    monte_carlo_shortfall,
    train_q_learning,
    twap_trajectory,
)

X, T, N, S0 = 10_000, 1.0, 10, 50.0
SIGMA, ETA, GAMMA, LAM = 0.3, 2.5e-6, 2.5e-7, 1e-4
N_LOTS = 20

env = ExecutionEnv(X=X, T=T, N=N, S0=S0, sigma=SIGMA, eta=ETA, gamma=GAMMA, lam=LAM, n_lots=N_LOTS, seed=0)
Q = train_q_learning(env, n_episodes=200_000, alpha=0.05, epsilon_decay_episodes=150_000, seed=1)
rl_traj = extract_policy_trajectory(Q, N, N_LOTS) * env.lot_size

ac_traj = ac_trajectory(X, T, N, SIGMA, ETA, LAM)
twap_traj = twap_trajectory(X, N)

print("Inventory remaining trajectory (shares):")
print(f"{'step':>5} {'RL':>10} {'AC closed-form':>16} {'TWAP':>10}")
for k in range(N + 1):
    print(f"{k:>5} {rl_traj[k]:>10.0f} {ac_traj[k]:>16.0f} {twap_traj[k]:>10.0f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(N + 1), rl_traj, "o-", label="Q-learning (RL)")
ax.plot(range(N + 1), ac_traj, "s--", label="Almgren-Chriss (closed form)")
ax.plot(range(N + 1), twap_traj, "^:", label="TWAP (naive)")
ax.set_xlabel("Step")
ax.set_ylabel("Inventory remaining (shares)")
ax.set_title("Optimal execution: RL vs. closed-form vs. naive baseline")
ax.legend()
fig.tight_layout()
fig.savefig("execution_trajectories.png", dpi=150)

tau = T / N
schedules = {
    "RL": -np.diff(rl_traj),
    "Almgren-Chriss": -np.diff(ac_traj),
    "TWAP": -np.diff(twap_traj),
}

print("\nMonte Carlo realized implementation shortfall (5,000 sims):")
print(f"{'strategy':>16} {'mean ($)':>12} {'std ($)':>12}")
for name, sched in schedules.items():
    costs = monte_carlo_shortfall(sched, S0, SIGMA, tau, ETA, GAMMA, n_sims=5000, seed=42)
    print(f"{name:>16} {costs.mean():>12.1f} {costs.std():>12.1f}")
