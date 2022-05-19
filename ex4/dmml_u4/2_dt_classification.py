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


def get_unsure_test_items(clf, X_test: np.ndarray) -> np.ndarray:
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



def plot_decision_boundary(clf, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray, y_test_unsure: np.ndarray) -> None:
    """Create a decision boundary plot that shows the predicted label for each point."""
    raise NotImplementedError


def get_acc_per_max_depth(X: np.ndarray, y: np.ndarray, max_depths: range) -> [np.ndarray, np.ndarray]:
    """Runs 10-fold cross validation for all given depths and
     returns an array for the corresponding train and test accuracy"""
    raise NotImplementedError


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

    y_test_unsure = get_unsure_test_items(clf, X_test)

    plot_decision_boundary(clf, X_train, X_test, y_train, y_test, y_test_unsure)

    # Make experiments reproducible
    np.random.seed(0)

    # Define interval
    max_depths = range(1, 16)
    acc_train, acc_test = get_acc_per_max_depth(X, y, max_depths)

    utils.plot_results(acc_test, acc_train, max_depths)


if __name__ == '__main__':
    main()
