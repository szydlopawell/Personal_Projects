import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from volatility import yang_zhang_volatility

TICKER = "SPY"
WINDOW = 20

ohlc = yf.download(TICKER, period="2y", auto_adjust=False, progress=False)
ohlc.columns = ohlc.columns.get_level_values(0)  # drop yfinance's ticker sub-level

yz_vol = yang_zhang_volatility(ohlc, window=WINDOW)

close_to_close = np.log(ohlc["Close"] / ohlc["Close"].shift(1))
cc_vol = close_to_close.rolling(WINDOW).std() * np.sqrt(252)

fig, ax = plt.subplots(figsize=(10, 5))
yz_vol.plot(ax=ax, label="Yang-Zhang")
cc_vol.plot(ax=ax, label="Close-to-close")
ax.set_title(f"{TICKER}: {WINDOW}-day annualized volatility")
ax.set_ylabel("Annualized volatility")
ax.legend()
fig.tight_layout()
fig.savefig("yang_zhang_demo.png", dpi=150)

print(yz_vol.dropna().tail())
