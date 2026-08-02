"""Walk-forward evaluation of the ML model against the naive baselines.

Uses sklearn's TimeSeriesSplit rather than random K-fold: each fold trains
only on data strictly before the test fold, so the model is never evaluated
on a period it could have (even indirectly, via shuffled folds) learned
from. This is the standard way to avoid the lookahead bias that plagues
naive backtests of return/volatility prediction models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def walk_forward_evaluate(X: pd.DataFrame, y: pd.Series, naive: pd.Series, ewma: pd.Series,
                           n_splits: int = 5, seed: int = 0):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=seed)
        model.fit(X_train, y_train)
        ml_pred = model.predict(X_test)

        rows.append({
            "fold": fold,
            "test_size": len(test_idx),
            "rmse_ml": rmse(y_test, ml_pred),
            "mae_ml": mean_absolute_error(y_test, ml_pred),
            "rmse_naive": rmse(y_test, naive.loc[y_test.index]),
            "mae_naive": mean_absolute_error(y_test, naive.loc[y_test.index]),
            "rmse_ewma": rmse(y_test, ewma.loc[y_test.index]),
            "mae_ewma": mean_absolute_error(y_test, ewma.loc[y_test.index]),
        })

    return pd.DataFrame(rows)


def feature_importances(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.Series:
    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=seed)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
