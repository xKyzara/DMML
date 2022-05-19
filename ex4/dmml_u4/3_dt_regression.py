#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import utils


def get_dt_prediction(X_train: np.ndarray, y_train: np.ndarray, max_depth: int) -> np.ndarray:
    """Fits a decision tree regressor and returns its predictions as an array."""
    regr = DecisionTreeRegressor(max_depth)
    regr.fit(X_train, y_train)
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    return regr.predict(X_test)



def plot_regression_models(X_cont: np.ndarray, y_cont: np.ndarray, y_1: np.ndarray, y_2: np.ndarray) -> None:
    """Plots the regression results for y_1 and y_2 on top of the training data X_cont, y_cont."""
    raise NotImplementedError


def main():
    np.random.seed(42)
    # create dataset
    X_train, y_train = utils.distorted_sin()
    utils.plot_points(X_train, y_train)

    y_1, y_2 = [get_dt_prediction(X_train, y_train, max_depth) for max_depth in [2, 5]]

    # Plot the results
    plot_regression_models(X_train, y_train, y_1, y_2)


if __name__ == '__main__':
    main()
