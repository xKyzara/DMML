#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import utils


def run_grid_search(X_digits, y_digits, C_range, degree_range, gamma_range):
    """Defines and runs a grid search over the given hyperparameter ranges for
    rbf and polynomial kernel. Each run is evaluated by using a 5-fold-cross-validation."""
    # Define grid over parameters for rbf and poly kernel
    rbf_grid = {"kernel": ["rbf"], "C": C_range, "gamma": gamma_range}
    poly_grid = {"kernel": ["poly"], "C": C_range, "degree": degree_range}
    # GridSearch arguments
    args = dict(estimator = svm.SVC(), scoring = make_scorer(accuracy_score), cv = 5, verbose = 1, n_jobs = -1)
    # Run gridsearch on the RBF grid
    rbf_gs = GridSearchCV(param_grid=rbf_grid, **args).fit(X_digits, y_digits)
    poly_gs = GridSearchCV(param_grid=poly_grid, **args).fit(X_digits, y_digits)
    return poly_gs, rbf_gs


def main():
    # Load digits
    X_digits, y_digits = datasets.load_digits(return_X_y=True)

    # Show data
    utils.plot_digits(X_digits, title="Digits")

    # Define hyperparameter ranges
    gamma_range = [2 ** i for i in range(-13, -4)]
    C_range = [2 ** i for i in range(-3, 4)]
    degree_range = range(1, 7)

    poly_gs, rbf_gs = run_grid_search(X_digits, y_digits, C_range, degree_range, gamma_range)

    # Visualize the gridsearch results
    plt.figure(figsize=(8, 4))
    plt.suptitle("GridSearch Validation Accuracy")
    utils.visualize_gridsearch(
        poly_gs,
        C_range,
        degree_range,
        ylabel="C",
        xlabel="degree",
        title="SVM Poly Kernel",
        fignum=121,
    )
    utils.visualize_gridsearch(
        rbf_gs,
        C_range,
        gamma_range,
        ylabel="C",
        xlabel="$\gamma$",
        title="SVM RBF Kernel",
        logx=True,
        fignum=122,
        cbar=True,
    )
    plt.show()


if __name__ == '__main__':
    main()
