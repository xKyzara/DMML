#!/usr/bin/env python
# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import utils


def map_polynomial(x: np.ndarray, degree: int) -> np.ndarray:
    """Create polynomial transformation of degree d:
     Maps each datapoint x_i, x_(i+1), ..., x_k to its polynomial:
      [x_i, x_(i+1), ..., x_k] -> [[x_i^0,     ..., x_i^d],
                                   [x_(i+1)^0, ..., x_(i+1)^d],
                                               ....
                                   [x_k^0,     ..., x_k^d]]."""
    x_poly = []
    # For each datapoint
    for xi in x.reshape(-1):
        xi_poly = [xi ** d for d in range(degree + 1)]
        x_poly.append(xi_poly)
    return np.array(x_poly)


def plot_polynomial_fits(x_test: np.ndarray, x_train: np.ndarray, y_test: np.ndarray, y_train: np.ndarray) -> None:
    """Plots regression curves for varying degrees."""
    # Plot different polynomial fits
    plt.figure(figsize=(6, 4))
    for i, d in enumerate([0, 1, 5, 9]):
        plt.subplot(2, 2, i + 1)

        # Evaluate polynomial regression with degree d
        error, y_pred = polynomial_regression(d, x_test, x_train, y_test, y_train)

        # Plot fit
        utils.plot_polyfit(x_train, y_train, x_test, y_test, y_pred, d, error)
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, ncol=1)
    plt.tight_layout()
    plt.show()


def polynomial_regression(d: int, x_test: np.ndarray, x_train: np.ndarray,
                          y_test: np.ndarray, y_train: np.ndarray) -> (float, np.ndarray):
    """Runs a polynomial regression of degree d and returns the test mean squared error and test y predictions."""
    # Map train and test set to polynomial space of degree d
    x_train_poly = map_polynomial(x_train, d)
    x_test_poly = map_polynomial(x_test, d)
    # Create a linear regression model in the polynomial features space
    model = LinearRegression()
    model.fit(x_train_poly, y_train)
    # Obtain predictions
    y_pred = model.predict(x_test_poly)
    # Evaluate predictions
    test_error = mse(y_test, y_pred)
    return test_error, y_pred


def main():
    # Generate data
    x_train = np.linspace(0, 1, 10).reshape(-1, 1)
    y_train = utils.fsin(x_train) + np.random.normal(scale=0.25, size=x_train.shape)
    x_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_test = utils.fsin(x_test)

    utils.plot_sin_curve(x_test, x_train, y_test, y_train)
    plot_polynomial_fits(x_test, x_train, y_test, y_train)


if __name__ == '__main__':
    main()
