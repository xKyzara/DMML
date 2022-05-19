from matplotlib import pyplot as plt
import numpy as np


def plot_data(X, y):
    plt.scatter(*X[y == 0].T, label="Class A")
    plt.scatter(*X[y == 1].T, label="Class B")
    plt.xlabel("$X_0$")
    plt.ylabel("$X_1$")


def plot_decision_regions(X, predict_fn):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid = np.c_[XX.ravel(), YY.ravel()]
    Z = np.array([predict_fn(x_i) for x_i in grid])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)


def plot_data_with_wrong_predictions(X, y, preds, predict_fn):
    plot_decision_regions(X, predict_fn)

    # Data
    plot_data(X, y)

    # Wrong prediction
    plt.scatter(
        *X[y != preds].T,
        edgecolor="red",
        facecolor="none",
        marker="X",
        zorder=1,
        s=80,
        label="Wrong pred.",
    )


def predict_from_boundary(x, boundary):
    # Find the two boundary points where x is in based on x_0 value
    interval_start, interval_end = None, None
    for i in np.arange(start=0, stop=len(boundary) - 1, step=1):
        if boundary[i][0] <= x[0] < boundary[i + 1][0]:
            interval_start, interval_end = boundary[i], boundary[i + 1]
            break
    assert (
        interval_start is not None and interval_end is not None
    ), f"No interval found for point {x}"

    # Compute slope and intercept
    m = (interval_end[1] - interval_start[1]) / (
        interval_end[0] - interval_start[0]
    )  # m = y2 - y1 / x2 - x1
    b = interval_end[1] - m * interval_end[0]  # b = y - mx

    # Classify based on whether point is above or below line
    if x[1] > m * x[0] + b:
        return 0
    else:
        return 1


from functools import partial
from sklearn.metrics import accuracy_score


def plot_data_with_custom_boundary(X, y, boundary):
    # Sort boundary points according to x_0 value
    boundary_sorted = sorted(boundary, key=lambda k: k[0])

    y_pred = [predict_from_boundary(x_i, boundary_sorted) for x_i in X]
    acc = accuracy_score(y, y_pred)
    plot_data_with_wrong_predictions(
        X,
        y,
        y_pred,
        predict_fn=partial(predict_from_boundary, boundary=boundary_sorted),
    )
    plt.plot(*np.array(boundary_sorted).T, linestyle="--", label="Boundary", c="black")
    plt.title(f"Accuracy: {acc * 100:2.2f}%")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))

    plt.legend()


def predict_all(X: np.ndarray, predict_fn) -> np.ndarray:
    """Predict all elements in the given array with the given prediction function."""
    return np.array([predict_fn(x_i) for x_i in X])
