import numpy as np
import pytest

from pricing import black_scholes_price

S0, K, r, T, sigma = 100.0, 100.0, 0.04, 1.0, 0.2


def test_put_call_parity():
    call = black_scholes_price(S0, K, r, T, sigma, option_type="call")
    put = black_scholes_price(S0, K, r, T, sigma, option_type="put")
    assert call - put == pytest.approx(S0 - K * np.exp(-r * T))


def test_call_price_is_nonnegative():
    price = black_scholes_price(S0, K, r, T, sigma, option_type="call")
    assert price >= 0


def test_put_price_is_nonnegative():
    price = black_scholes_price(S0, K, r, T, sigma, option_type="put")
    assert price >= 0


def test_call_approaches_intrinsic_value_as_vol_to_zero():
    tiny_sigma = 1e-6
    price = black_scholes_price(S0, K, r, T, tiny_sigma, option_type="call")
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    assert abs(price - intrinsic) < 1e-3


def test_deep_itm_call_approaches_forward_minus_strike():
    deep_itm_S0 = 1000.0
    price = black_scholes_price(deep_itm_S0, K, r, T, sigma, option_type="call")
    forward_value = deep_itm_S0 - K * np.exp(-r * T)
    assert abs(price - forward_value) < 1.0


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        black_scholes_price(S0, K, r, T, sigma, option_type="invalid")
