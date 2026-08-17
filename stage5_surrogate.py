"""
stage5_surrogate.py - Stage V: surrogate training and Top-K Gain validation.

Trains a lightweight XGBoost regressor that maps the 26-dimensional zero-cost
proxy features (Stage II) to the noise-aware performance labels produced by the
Stage IV evaluation pipeline, and validates the ranking capability of the
trained predictor with the Top-K Gain AUC criterion implemented in
compute_metric.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from compute_metric import compute_relative_topk_gain_auc

# Reasonable default hyperparameters for the surrogate regressor.
DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 4,
    "gamma": 1e-7,
}


def train_surrogate(X, y, params: dict = None, seed: int = 42) -> xgb.XGBRegressor:
    """Train the XGBoost surrogate on (proxy features, performance labels)."""
    params = dict(DEFAULT_PARAMS if params is None else params)
    model = xgb.XGBRegressor(
        n_estimators=params.pop("n_estimators", DEFAULT_PARAMS["n_estimators"]),
        max_depth=params.pop("max_depth", DEFAULT_PARAMS["max_depth"]),
        learning_rate=params.pop("learning_rate", DEFAULT_PARAMS["learning_rate"]),
        subsample=params.pop("subsample", DEFAULT_PARAMS["subsample"]),
        colsample_bytree=params.pop("colsample_bytree", DEFAULT_PARAMS["colsample_bytree"]),
        min_child_weight=params.pop("min_child_weight", DEFAULT_PARAMS["min_child_weight"]),
        gamma=params.pop("gamma", DEFAULT_PARAMS["gamma"]),
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def validate_relative_topk_gain(
    model: xgb.XGBRegressor,
    X_test,
    y_test,
    exp_decay_rate: float = 5.0,
) -> dict:
    """Rank test architectures by surrogate predictions and compute the Top-K Gain AUC.

    Uses compute_relative_topk_gain_auc (the criterion defined in the
    manuscript). Returns the normalized mean/max Top-K Gain AUC (higher is
    better). Scores are assumed to be "larger is better"; negate VQE energies
    before calling this function.
    """
    y_pred = np.asarray(model.predict(X_test))
    y_true = np.asarray(y_test)
    mean_auc, max_auc = compute_relative_topk_gain_auc(
        y_pred, y_true, exp_decay_rate=exp_decay_rate
    )
    return {
        "topk_gain_auc_mean": float(mean_auc),
        "topk_gain_auc_max": float(max_auc),
    }


def train_and_validate(
    X,
    y,
    params: dict = None,
    test_size: float = 0.2,
    seed: int = 42,
    exp_decay_rate: float = 5.0,
):
    """Train-test split, fit the surrogate, and report Top-K Gain + Spearman."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    model = train_surrogate(X_train, y_train, params=params, seed=seed)
    metrics = validate_relative_topk_gain(
        model, X_test, y_test, exp_decay_rate=exp_decay_rate
    )
    metrics["spearman"] = float(spearmanr(model.predict(X_test), y_test).correlation)
    return model, metrics


def _synthetic_demo(seed: int = 42, n_samples: int = 300) -> None:
    """Self-contained illustration with synthetic proxy/label data."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n_samples, 26)),
        columns=[f"proxy_{i}" for i in range(26)],
    )
    y = (
        0.5 * X["proxy_0"]
        + 0.3 * X["proxy_1"]
        - 0.4 * X["proxy_2"]
        + 0.1 * rng.standard_normal(n_samples)
    ).to_numpy()
    model, metrics = train_and_validate(X, y, seed=seed)
    print(f"Top-K Gain AUC (mean/max): {metrics['topk_gain_auc_mean']:.4f} / "
          f"{metrics['topk_gain_auc_max']:.4f}")
    print(f"Spearman correlation      : {metrics['spearman']:.4f}")


if __name__ == "__main__":
    _synthetic_demo()
