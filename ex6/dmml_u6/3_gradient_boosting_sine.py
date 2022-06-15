# coding: utf-8
# - https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
# [1] J. Friedman, Greedy Function Approximation: A Gradient Boosting Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
# [2] T. Hastie, R. Tibshirani and J. Friedman, The Elements of Statistical Learning, Second Edition, Section 10.13.2, Springer, 2009.
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
import utils


def fit_gradient_boosting_regressor(X: np.ndarray, n_samples: int, y: np.ndarray,
                                    learning_rate: float, max_depth: int, random_state: int, loss: str):
    """Fits gradient tree boosting regressor and returns the learned model and training errors."""
    raise NotImplementedError


def main():
    # Create sine function
    n_samples = 300
    X = np.linspace(0, 4 * np.pi, num=n_samples).reshape(-1, 1)
    y = np.sin(X[:, 0])

    # Plot sine function
    utils.plot_sine(X, y)

    gb, errors = fit_gradient_boosting_regressor(X, n_samples, y, learning_rate=0.5, max_depth=1,
                                                 random_state=0, loss="ls")

    # Plot against iterations
    utils.plot_loss(errors)

    # Obtain staged predictions, that is the predictions of the ensemble after each step
    staged_y_preds = [p for p in tqdm(gb.staged_predict(X))]

    # Show boosting iterations
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 15), dpi=300)
    axs = axs.flatten()
    step = 50
    for i in tqdm(range(0, n_samples, step)):
        utils.visualize_gradient_boosting_sine(X, y, axs, staged_y_preds, i=i, step=step)
    plt.show()


if __name__ == '__main__':
    main()
