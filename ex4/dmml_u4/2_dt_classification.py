#!/usr/bin/env python
# coding: utf-8
# references:
# https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py
# https://scikit-learn.org/stable/modules/tree.html

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import utils


def my_get_unsure_test_items(clf, X_test: np.ndarray) -> np.ndarray:
    """Returns a boolean array where all items with a probability != 1 are marked as true"""
    prob_arr = clf.predict_proba(X_test)
    bool_arr = []
    for prob in prob_arr:
        for p in prob:
            if p != 1:
                bool_arr.append(True)
            else:
                bool_arr.append(False)
    return bool_arr


def get_unsure_test_items(clf, X_test: np.ndarray) -> np.ndarray:
    """Returns a boolean array where all items with a probability != 1 are marked as true"""
    prob_arr = clf.predict_proba(X_test)
    return prob_arr.max(axis=1) != 1


def plot_decision_boundary(clf, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray, y_test_unsure: np.ndarray) -> None:
    """Create a decision boundary plot that shows the predicted label for each point."""
    cache = {}

    # Plot decision regions
    plt.figure()
    plt.title("Decision Tree Decision Regions on 2D Iris")

    h = 0.05  # step size in the mesh

    # Create color maps

    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])

    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


    k = clf.min_samples_leaf
    if k in cache:
        Z = cache[k]
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        cache[k] = Z

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training and test points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, marker="x", edgecolor="k", s=25, label="Training Points")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_train, cmap=cmap_bold, marker="x", edgecolor="k", s=25, label="Test Points")
    plt.scatter(X_test[y_test_unsure][:, 0], X_test[y_test_unsure][:, 1], c="yellow", marker="*", edgecolor="k", s=50, label="Unsure Points")

    plt.legend()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def get_acc_per_max_depth(X: np.ndarray, y: np.ndarray, max_depths: range) -> [np.ndarray, np.ndarray]:
    """Runs 10-fold cross validation for all given depths and
     returns an array for the corresponding train and test accuracy"""

    def evaluate(X, y, max_depth):
        model = tree.DecisionTreeClassifier(max_depth=max_depth)
        scores = cross_validate(estimator = model, X = X, y = y, scoring = "accuracy", cv = 10, return_train_score = True)
        return np.mean(scores["train_score"]), np.mean(scores["test_score"])

    # Evaluate each depth parameter
    accuracies = np.array([evaluate(X, y, md) for md in max_depths])
    acc_train = accuracies[:, 0]
    acc_test = accuracies[:, 1]
    return acc_train, acc_test


def main():
    # Load the iris dataset
    iris = load_iris()

    # Fit a Decision tree on the iris data
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)

    utils.plot_decision_tree(clf, iris)

    X, y = utils.reduce_iris_to_2d(iris)

    # Train decision tree
    clf = tree.DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = clf.fit(X_train, y_train)

    y_test_unsure = my_get_unsure_test_items(clf, X_test)

    plot_decision_boundary(clf, X_train, X_test, y_train, y_test, y_test_unsure)

    # Make experiments reproducible
    np.random.seed(0)

    # Define interval
    max_depths = range(1, 16)
    acc_train, acc_test = get_acc_per_max_depth(X, y, max_depths)

    utils.plot_results(acc_test, acc_train, max_depths)


if __name__ == '__main__':
    main()
