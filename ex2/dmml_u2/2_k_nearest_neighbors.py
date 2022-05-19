#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import utils


def load_reduced_iris_dataset() -> (np.ndarray, np.ndarray):
    """Loads the iris dataset reduced to its first two features (sepal width, sepal length)."""
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    return X, y


def evaluate_ks(ks: range, X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
    """Evaluates each k-value of the ks-array and returns their respective training and test accuracy."""
    accuracies = np.array([utils.evaluate(k, X, y) for k in ks])
    acc_train = accuracies[:, 0]
    acc_test = accuracies[:, 1]
    return acc_train, acc_test


def plot_k_to_acc(ks: range, acc_train: np.ndarray, acc_test: np.ndarray) -> None:
    """Plots the k-values in relation to their respective training and test accuracy."""
    plt.figure()
    plt.scatter(ks, acc_train, label="Accuracy Train", marker="x")
    plt.scatter(ks, acc_test, label="Accuracy Test", marker="^")
    plt.legend()
    plt.xlabel("$k$ Neighbors")
    plt.ylabel("Accuracy")
    plt.title("KNN: Accuracy vs. Number of Neighbors")
    plt.show()


def get_best_k(ks: range, acc_test: np.ndarray) -> int:
    """Returns the best value for k based on the highest test accuracy."""
    best_k = ks[acc_test.argmax()]
    return best_k


def plot_decision_boundary_for_k(k: int, X: np.ndarray, y: np.ndarray) -> None:
    """Creates and fits a KNN model with value k and plots its decision boundary."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    utils.plot_decision_boundary(knn, X, y)


def main():
    # Set seed to make experiments reproducible
    np.random.seed(0)
    X, y = load_reduced_iris_dataset()
    utils.plot_iris_dataset(X, y)

    # Define interval
    ks = range(1, 100, 2)
    acc_train, acc_test = evaluate_ks(ks, X, y)

    plot_k_to_acc(ks, acc_train, acc_test)

    best_k = get_best_k(ks, acc_test)
    print(f"best k value: {best_k}")

    for k in range(1, 101, 20):
        plot_decision_boundary_for_k(k, X, y)


if __name__ == '__main__':
    main()
