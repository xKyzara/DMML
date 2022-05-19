#!/usr/bin/env python
# coding: utf-8
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np
import utils


def create_blob_dataset() -> [np.ndarray, np.ndarray]:
    """Create blobs of independent Gaussian distributions with centers at (-1.5,-1.5) and (1.5,1.5)"""
    return datasets.make_blobs(n_samples=500, random_state=0, centers=[[-1.5, -1.5], [1.5, 1.5]])


def predict_1(x: np.ndarray) -> int:
    """Predict the class based on whether it is in the third quadrant."""
    if x[0] < 0 and x[1] < 0:
        return 0  # Class A
    else:
        return 1  # Class B


def predict_2(x: np.ndarray) -> int:
    """Predict the class based on whether it is below the line -x_0 = x_1."""
    if -x[0] > x[1]:
        return 0  # Class A
    else:
        return 1  # Class B


def accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the accuracy of the prediction y_pred w.r.t. the true labels y."""
    num_correct = 0
    num_total = y.shape[0]
    for y_i, y_i_pred in zip(y, y_pred):
        # Count correct predictions
        if y_i == y_i_pred:
            num_correct += 1
        # Calculate accuracy
        return num_correct / num_total * 100


def accuracy2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the accuracy of the prediction y_pred w.r.t. the true labels y."""
    return (y == y_pred).sum() / len(y) * 100


def get_boundaries() -> list:
    """Returns a list of boundary points e.g.
    [[-5, 0],[-2, 2],[-1,1],[1, 0.9],[4, 2],]
    """
    return [[-5, 0], [-0.8, -0.5], [-0.1, 0.8], [1.2, -0.3], [5, 5]]


def main():
    sns.set(
        context="notebook",
        style="whitegrid",
        rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
    )

    # Create blobs of independent Gaussian distributions
    X, y = create_blob_dataset()

    # Plot
    plt.figure()
    utils.plot_data(X, y)
    plt.legend()
    plt.show()

    functions = [predict_1, predict_2]

    for predict in functions:
        # Predict all datapoints
        y_pred = utils.predict_all(X, predict)

        # Calculate the accuracy
        acc = accuracy(y, y_pred)
        print(f"Accuracy: {acc:.2f}%")

        # Show wrong predictions
        plt.figure()
        utils.plot_data_with_wrong_predictions(X, y, y_pred, predict_fn=predict)
        plt.legend()
        plt.show()

    # Load data
    X_moons, y_moons = datasets.make_moons(noise=0.2, random_state=0)

    # Hand-pick decision boundary points
    # The boundary is defined by connecting two successive points
    # Add [x_0, x_1] values to extend and improve the border
    boundary = get_boundaries()

    # Show data with decision boundary
    utils.plot_data_with_custom_boundary(X_moons, y_moons, boundary)
    plt.show()


if __name__ == "__main__":
    main()
