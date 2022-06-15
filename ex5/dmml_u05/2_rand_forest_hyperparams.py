#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.model_selection import cross_validate
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import seaborn as sns

sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
)


def evaluate_n_estimators(X: np.ndarray, y: np.ndarray, n: int) -> (float, float):
    """"Run 10 fold cross-validation of the model for a given number of trees and returns the
    mean train and test score."""
    model = RandomForestClassifier(n_estimators=n)
    scores = cross_validate(
        estimator=model, X=X, y=y, scoring="accuracy", cv=10, return_train_score=True, n_jobs=-1,
    )
    return np.mean(scores["train_score"]), np.mean(scores["test_score"])



def evaluate_depth(X: np.ndarray, y: np.ndarray, depth: int) -> (float, float):
    """Run 10 fold cross-validation of the model for a given tree depth and returns the
    mean train and test score."""
    model = RandomForestClassifier(max_depth=depth, n_estimators=20)
    scores = cross_validate(
        estimator = model, X = X, y = y, scoring = "accuracy", cv = 10, return_train_score = True, n_jobs = -1,
    )
    return np.mean(scores["train_score"]), np.mean(scores["test_score"])


def evaluate_features(X: np.ndarray, y: np.ndarray, n_features: int) -> (float, float):
    """"Run 10 fold cross-validation of the model for a given number of features per tree and returns the
    mean train and test score."""
    model = RandomForestClassifier(max_features=n_features, n_estimators=20)
    scores = cross_validate(
        estimator = model, X = X, y = y, scoring = "accuracy", cv = 10, return_train_score = True, n_jobs = -1,
    )
    return np.mean(scores["train_score"]), np.mean(scores["test_score"])


def plot_accuracy(xs: range, accuracies: np.ndarray, xlabel: str, ylabel="Accuracy") -> None:
    """Plot results for the given accuracies."""
    acc_train = accuracies[:, 0]
    acc_test = accuracies[:, 1]
    plt.figure()
    plt.plot(xs, acc_train, label="Train", linestyle="--")
    plt.plot(xs, acc_test, label="Test", linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xs[::5])
    plt.legend()
    plt.show()


def main():
    # Load digits
    X, y = load_digits(return_X_y=True)

    # Plot samples
    plt.figure()
    for i, (x_i, y_i) in enumerate(zip(X[:4], y[:4]), start=1):
        plt.subplot(140 + i)
        plt.imshow(x_i.reshape(8, 8), cmap="gray")
        plt.title("label = " + str(y_i))
        plt.axis("off")

    # Define interval
    n_estimators = range(1, 41)

    # Evaluate interval
    accuracies_n_est = np.array([evaluate_n_estimators(X, y, alpha) for alpha in tqdm(n_estimators)])

    plot_accuracy(n_estimators, accuracies_n_est, "Number of Trees")

    # Define interval
    depths = range(1, 15)

    # Evaluate interval
    accuracies_depths = np.array([evaluate_depth(X, y, d) for d in tqdm(depths)])

    # Plot results
    plot_accuracy(depths, accuracies_depths, "Tree Depth")

    # Define interval
    n_features = range(1, X.shape[1], 1)

    # Evaluate interval
    accuracies_n_feat = np.array([evaluate_features(X, y, n) for n in tqdm(n_features)])

    # Plot results
    plot_accuracy(n_features, accuracies_n_feat, "Max. Number of Features per Tree")


if __name__ == '__main__':
    main()
