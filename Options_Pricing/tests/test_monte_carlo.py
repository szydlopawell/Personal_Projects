from pricing import american_option_lsm, black_scholes_price, european_option_mc

S0, K, r, T, sigma = 100.0, 100.0, 0.04, 0.5, 0.25


def test_european_mc_matches_black_scholes_within_confidence_interval():
    bs_price = black_scholes_price(S0, K, r, T, sigma, option_type="put")
    mc_price, se = european_option_mc(S0, K, r, T, sigma, option_type="put",
                                       n_steps=50, n_paths=40_000, seed=1)
    assert abs(mc_price - bs_price) < 3 * se


def test_american_put_worth_at_least_as_much_as_european():
    # An American option can never be worth less than its European counterpart --
    # it has every European exercise opportunity plus the option to exercise early.
    euro_price, _ = european_option_mc(S0, K, r, T, sigma, option_type="put",
                                        n_steps=50, n_paths=40_000, seed=2)
    amer_price, _ = american_option_lsm(S0, K, r, T, sigma, option_type="put",
                                         n_steps=50, n_paths=40_000, seed=2)
    assert amer_price >= euro_price - 1e-6


def test_american_call_without_dividends_equals_european():
    # With no dividends, early exercise of a call is never optimal, so the
    # LSM price should match the European (terminal-payoff-only) price.
    euro_price, euro_se = european_option_mc(S0, K, r, T, sigma, option_type="call",
                                              n_steps=50, n_paths=40_000, seed=3)
    amer_price, _ = american_option_lsm(S0, K, r, T, sigma, option_type="call",
                                         n_steps=50, n_paths=40_000, seed=3)
    assert abs(amer_price - euro_price) < 3 * euro_se


def test_antithetic_reduces_standard_error():
    _, se_plain = european_option_mc(S0, K, r, T, sigma, option_type="put",
                                      n_steps=50, n_paths=10_000, seed=4, antithetic=False)
    _, se_antithetic = european_option_mc(S0, K, r, T, sigma, option_type="put",
                                           n_steps=50, n_paths=10_000, seed=4, antithetic=True)
    assert se_antithetic < se_plain


def test_zero_strike_call_worth_full_stock_price():
    price, se = european_option_mc(S0, K=1e-6, r=r, T=T, sigma=sigma, option_type="call",
                                    n_steps=10, n_paths=20_000, seed=5)
    assert abs(price - S0) < 3 * se + 1e-3
