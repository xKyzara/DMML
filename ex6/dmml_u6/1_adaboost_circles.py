# coding: utf-8
# Referenzen
# - https://scikit-learn.org/stable/modules/ensemble.html#adaboost
# - https://xavierbourretsicotte.github.io/AdaBoost.html
# - [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical Learning Ed. 2", Springer, 2009.
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utils


def plot_dataset(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Creates as scatterplot for the given dataset. The points for the first class are blue and the points for the
     second class are red. Training points are displayed as dots and testpoints are described by an X."""
    plt.figure(figsize=(4, 4))
    plt.scatter(*X_train.T, c=y_train, cmap=ListedColormap(["#FF0000", "#0000FF"]), label="Training Points", marker="o")
    plt.scatter(*X_test.T, c=y_test, cmap=ListedColormap(["#FF0000", "#0000FF"]), label="Test Points", marker="x")
    plt.xlabel("$X_0$")
    plt.ylabel("$X_1$")
    plt.legend()
    plt.show()


def plot_decision_boundary_stump(stump: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray, N=1000) -> None:
    """Plot the decision boundary for a tree stump and scatters plot of the training data"""
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
    plt.figure(figsize=(4, 4))
    plt.title(f"Decision Boundary of a Decision Stump")
    zz = stump.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = zz[:, 1].reshape(xx.shape)
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    # Get current axis and plot
    ax = plt.gca()
    ax.contourf(xx, yy, Z, 2, cmap="RdBu", alpha=0.5)
    ax.contour(xx, yy, Z, 2, cmap="RdBu")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    ax.set_xlabel("$X_0$")
    ax.set_ylabel("$X_1$")


def main():
    # Generate data
    X, y = datasets.make_circles(noise=0.1, factor=0.4, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    # Plot data
    plot_dataset(X_train, X_test, y_train, y_test)

    decision_stump = DecisionTreeClassifier(max_depth=1)
    decision_stump.fit(X_train, y_train)

    # Plot the decision boundary
    plt.figure()
    plot_decision_boundary_stump(decision_stump, X_train, y_train)
    plt.show()

    # Create AdaBoost model with decision stump as base learner
    boost = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=4,
        learning_rate=0.1,
    )

    # Fit model to the data
    boost.fit(X, y)
    plt.figure(figsize=(2 * 4, boost.n_estimators * 4))
    utils.plot_decision_boundary_adaboost(boost, X, y)
    plt.savefig("adaboost_progress.pdf")
    plt.show()


if __name__ == '__main__':
    main()
