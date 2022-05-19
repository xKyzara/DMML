import numpy as np
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
)  # Reenable seaborn style


def plot_decision_tree(clf, iris):
    plt.figure(figsize=(10, 6), dpi=160)
    sns.reset_orig()  # (Hack to make seaborn not modify the tree plot)
    tree.plot_tree(clf.fit(iris.data, iris.target))
    plt.show()


def reduce_iris_to_2d(iris):
    X = iris.data[:, :2]
    y = iris.target
    return X, y


def plot_results(acc_test, acc_train, depths_to_eval):
    plt.figure()
    plt.scatter(depths_to_eval, acc_test, label="Accuracy Test", marker="^")
    plt.scatter(depths_to_eval, acc_train, label="Accuracy Train", marker="x")
    plt.legend()
    plt.xlabel("Maximum Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree: Accuracy vs. Max. Depth")
    plt.show()


def distorted_sin():
    # Create a random dataset
    X_cont = np.sort(10 * np.random.rand(80, 1), axis=0)
    y_cont = np.sin(X_cont).ravel()
    y_cont += 0.5 * (0.5 - np.random.rand(80))
    y_cont[::5] += 3 * (0.5 - np.random.rand(16))
    return X_cont, y_cont


def plot_points(X_cont, y_cont):
    plt.figure()
    plt.title("Noisy Sine")
    plt.scatter(X_cont, y_cont)
    plt.show()
