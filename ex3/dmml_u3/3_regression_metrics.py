#!/usr/bin/env python
# coding: utf-8
# references:
# - https://scikit-learn.org/stable/modules/model_evaluation.html
# - https://developers.google.com/machine-learning/crash-course/classification

import numpy as np
import seaborn as sns
import sklearn
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)


sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
)


def evaluate_metric(X: np.ndarray, y: np.ndarray, clf: sklearn.base.BaseEstimator, metric: staticmethod.__func__)\
        -> float:
    """Runs a 5-fold cross validation and evaluates the given metric error function."""
    y_pred = cross_val_predict(estimator=clf, X=X, y=y, cv=5)
    err = metric(y, y_pred)
    return err


def main():
    # Load dataset
    X, y = datasets.load_boston(return_X_y=True)
    clf = RandomForestRegressor()

    # Evaluate predictions w.r.t. the above metrics
    print(f"{'Metric':<24}{'Score':>6}")
    print("-" * 30)
    for metric, name in [
        (mean_absolute_error, "Mean Absolute Error"),
        (mean_squared_error, "Mean Squared Error"),
        (
            lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "Root Mean Squared Error",
        ),
        (mean_squared_log_error, "Mean Log Squared Error"),
        (median_absolute_error, "Median Absolute Error"),
        (r2_score, "R2 Score"),
    ]:

        err = evaluate_metric(X, y, clf, metric)
        print(f"{name:<24}{err: >6.2f}")


if __name__ == '__main__':
    main()
