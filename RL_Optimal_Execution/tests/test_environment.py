from execution.environment import ExecutionEnv, execution_price, shortfall_cost, update_price


def test_execution_price_no_impact_when_n_zero():
    assert execution_price(price=50.0, n=0, tau=0.1, eta=1e-5) == 50.0


def test_execution_price_decreases_with_larger_sell_size():
    fast = execution_price(price=50.0, n=1000, tau=0.1, eta=1e-5)
    slow = execution_price(price=50.0, n=100, tau=0.1, eta=1e-5)
    assert fast < slow < 50.0


def test_shortfall_cost_zero_when_filled_at_arrival_price():
    assert shortfall_cost(S0=50.0, n=100, exec_price=50.0) == 0.0


def test_shortfall_cost_positive_when_filled_below_arrival_price():
    assert shortfall_cost(S0=50.0, n=100, exec_price=49.0) > 0


def test_update_price_no_shock_no_impact_is_unchanged():
    assert update_price(price=50.0, n=0, sigma=0.3, tau=0.1, gamma=1e-6, z=0.0) == 50.0


def test_update_price_permanent_impact_pushes_price_down_when_selling():
    price = update_price(price=50.0, n=1000, sigma=0.0, tau=0.1, gamma=1e-6, z=0.0)
    assert price < 50.0


def test_reset_returns_full_inventory_at_step_zero():
    env = ExecutionEnv(X=10_000, N=10, n_lots=20, seed=0)
    state = env.reset()
    assert state == (0, 20)


def test_step_forces_full_liquidation_on_last_step():
    env = ExecutionEnv(X=10_000, N=3, n_lots=20, seed=0)
    env.reset()
    env.step(0)  # step 0 -> 1: sell nothing
    env.step(0)  # step 1 -> 2: sell nothing, now at the last step (k == N-1)
    (k, inv), _, done, _ = env.step(0)  # agent asks for 0, env should force full liquidation
    assert inv == 0
    assert done


def test_step_action_is_clamped_to_available_inventory():
    env = ExecutionEnv(X=10_000, N=10, n_lots=20, seed=0)
    env.reset()
    (k, inv), _, _, _ = env.step(sell_lots=999)  # can't sell more than the 20 lots held
    assert inv == 0


def test_episode_ends_after_n_steps():
    N = 5
    env = ExecutionEnv(X=10_000, N=N, n_lots=20, seed=0)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(0)
        steps += 1
    assert steps == N
