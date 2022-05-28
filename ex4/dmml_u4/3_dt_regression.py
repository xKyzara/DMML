#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import utils


def get_dt_prediction(X_train: np.ndarray, y_train: np.ndarray, max_depth: int) -> np.ndarray:
    """Fits a decision tree regressor and returns its predictions as an array."""
    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=max_depth)
    regr_1.fit(X_train, y_train)
    # Predict
    y_pred = regr_1.predict(X_train)
    return y_pred



def plot_regression_models(X_cont: np.ndarray, y_cont: np.ndarray, y_1: np.ndarray, y_2: np.ndarray) -> None:
    """Plots the regression results for y_1 and y_2 on top of the training data X_cont, y_cont."""
    plt.figure(figsize=(12, 4))
    ax = plt.subplot(121)
    plt.scatter(X_cont, y_cont, s=20, edgecolor="black")
    plt.plot(X_cont, y_1, label="$max\_depth=2$", color="orange")
    ax.legend()
    ax.set_ylabel("Target")
    ax.set_xlabel("Data")
    ax = plt.subplot(122)
    plt.scatter(X_cont, y_cont, s=20, edgecolor="black")
    plt.plot(X_cont, y_2, label="$max\_depth=5$", color="green")
    ax.legend()
    plt.xlabel("Data")
    plt.suptitle("Decision Tree Regression")
    plt.show()


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
