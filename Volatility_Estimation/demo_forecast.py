import yfinance as yf

from forecasting import (
    build_features,
    ewma_forecast,
    feature_importances,
    naive_persistence_forecast,
    train_test_columns,
    walk_forward_evaluate,
)

TICKER = "SPY"
HORIZON = 20  # forecast the next 20 trading days' realized vol

ohlc = yf.download(TICKER, period="5y", auto_adjust=False, progress=False)
ohlc.columns = ohlc.columns.get_level_values(0)

data = build_features(ohlc, windows=(5, 10, 20, 60), horizon=HORIZON)
X, y = train_test_columns(data)

naive = naive_persistence_forecast(data)
ewma = ewma_forecast(ohlc, span=20).loc[data.index]

results = walk_forward_evaluate(X, y, naive, ewma, n_splits=5)
print(f"{TICKER}: {len(data)} rows, {HORIZON}-day-ahead realized vol forecast")
print("\nPer-fold walk-forward results (annualized vol units):")
print(results.to_string(index=False))

print("\nMean across folds:")
means = results[["rmse_ml", "rmse_naive", "rmse_ewma", "mae_ml", "mae_naive", "mae_ewma"]].mean()
print(means.to_string())

print("\nGradient Boosting feature importances (fit on full sample, for inspection only):")
print(feature_importances(X, y, naive, ewma).to_string())
