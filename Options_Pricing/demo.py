import pathlib
import sys

import yfinance as yf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "Volatility_Estimation"))
from volatility import yang_zhang_volatility  # noqa: E402

from pricing import american_option_lsm, black_scholes_price, european_option_mc  # noqa: E402

TICKER = "SPY"
OPTION_TYPE = "put"
R = 0.04  # flat risk-free rate proxy
T = 0.5   # 6 months to maturity
N_PATHS = 50_000
N_STEPS = 50
SEED = 42

ohlc = yf.download(TICKER, period="1y", auto_adjust=False, progress=False)
ohlc.columns = ohlc.columns.get_level_values(0)

sigma = yang_zhang_volatility(ohlc, window=20).dropna().iloc[-1]
S0 = float(ohlc["Close"].iloc[-1])
K = S0  # at-the-money

print(f"{TICKER}: S0={S0:.2f}, sigma (Yang-Zhang, 20d)={sigma:.2%}, K={K:.2f}, T={T}y, r={R:.2%}\n")

bs = black_scholes_price(S0, K, R, T, sigma, option_type=OPTION_TYPE)
euro_price, euro_se = european_option_mc(S0, K, R, T, sigma, option_type=OPTION_TYPE, n_steps=N_STEPS, n_paths=N_PATHS, seed=SEED)
amer_price, amer_se = american_option_lsm(S0, K, R, T, sigma, option_type=OPTION_TYPE, n_steps=N_STEPS, n_paths=N_PATHS, seed=SEED)

print(f"European {OPTION_TYPE} (Black-Scholes):  {bs:.4f}")
print(f"European {OPTION_TYPE} (Monte Carlo):    {euro_price:.4f}  (95% CI +/- {1.96*euro_se:.4f})")
print(f"American {OPTION_TYPE} (LSM):            {amer_price:.4f}  (95% CI +/- {1.96*amer_se:.4f})")
print(f"Early exercise premium:              {amer_price - euro_price:.4f}")
