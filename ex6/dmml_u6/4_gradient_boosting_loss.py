# coding: utf-8
# - https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
# [1] J. Friedman, Greedy Function Approximation: A Gradient Boosting Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
# [2] T. Hastie, R. Tibshirani and J. Friedman, The Elements of Statistical Learning, Second Edition, Section 10.13.2, Springer, 2009.
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import utils


def evaluate_loss_fn(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     loss_fn="ls", learning_rate=0.5) -> np.ndarray:
    """Fits a GradientBoostingRegressor on the training data and returns the training and test mse for all iterations"""
    raise NotImplementedError


def main():
    # Load boston regression
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Set up plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].set_title("Train MSE")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("MSE")
    ax[1].set_title("Test MSE")
    ax[1].set_xlabel("Iteration")
    ax[0].set_ylim(0, 40)
    ax[1].set_ylim(0, 40)

    # Run on all available loss functions
    for loss_fn in tqdm(["ls", "lad", "huber", "quantile"]):
        mses = evaluate_loss_fn(X_train, y_train, X_test, y_test, loss_fn)
        ax[0].plot(mses[:, 0], label=loss_fn)
        ax[1].plot(mses[:, 1], label=loss_fn)

    # Plot
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Run on a set of different learning rates
    for lr in [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]:
        mses = evaluate_loss_fn(X_train, y_train, X_test, y_test, learning_rate=lr, loss_fn="ls")
        ax[0].plot(mses[:, 0], label=lr)
        ax[1].plot(mses[:, 1], label=lr)

    # Set up plot
    utils.annotate_axes_lr(ax)


if __name__ == '__main__':
    main()
