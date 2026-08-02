"""Walk-forward evaluation of the ML model against the naive baselines.

Uses sklearn's TimeSeriesSplit rather than random K-fold: each fold trains
only on data strictly before the test fold, so the model is never evaluated
on a period it could have (even indirectly, via shuffled folds) learned
from. This is the standard way to avoid the lookahead bias that plagues
naive backtests of return/volatility prediction models.

The model itself is a blend of a Gradient Boosting regressor (trained on
log-volatility, since realized vol is right-skewed) and the EWMA baseline.
Giving the model the baseline forecasts as input features -- and then
blending its output back with EWMA -- lets it focus on learning small
corrections on top of a strong, already-good prior instead of relearning
volatility's persistence from scratch, which a plain regressor struggles to
beat on a dataset this size.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

BLEND_WEIGHT = 0.5  # weight on the model's own prediction; remainder goes to EWMA


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _make_model(seed):
    return GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=seed)


def _add_baseline_features(X: pd.DataFrame, naive: pd.Series, ewma: pd.Series) -> pd.DataFrame:
    X = X.copy()
    X["naive_forecast"] = naive.loc[X.index]
    X["ewma_forecast"] = ewma.loc[X.index]
    return X


def walk_forward_evaluate(X: pd.DataFrame, y: pd.Series, naive: pd.Series, ewma: pd.Series,
                           n_splits: int = 5, seed: int = 0):
    X_aug = _add_baseline_features(X, naive, ewma)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_aug), start=1):
        X_train, X_test = X_aug.iloc[train_idx], X_aug.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = _make_model(seed)
        model.fit(X_train, np.log(y_train))
        model_pred = np.exp(model.predict(X_test))
        ewma_test = ewma.loc[y_test.index]
        ml_pred = BLEND_WEIGHT * model_pred + (1 - BLEND_WEIGHT) * ewma_test

        rows.append({
            "fold": fold,
            "test_size": len(test_idx),
            "rmse_ml": rmse(y_test, ml_pred),
            "mae_ml": mean_absolute_error(y_test, ml_pred),
            "rmse_naive": rmse(y_test, naive.loc[y_test.index]),
            "mae_naive": mean_absolute_error(y_test, naive.loc[y_test.index]),
            "rmse_ewma": rmse(y_test, ewma_test),
            "mae_ewma": mean_absolute_error(y_test, ewma_test),
        })

    return pd.DataFrame(rows)


def feature_importances(X: pd.DataFrame, y: pd.Series, naive: pd.Series, ewma: pd.Series, seed: int = 0) -> pd.Series:
    X_aug = _add_baseline_features(X, naive, ewma)
    model = _make_model(seed)
    model.fit(X_aug, np.log(y))
    return pd.Series(model.feature_importances_, index=X_aug.columns).sort_values(ascending=False)
