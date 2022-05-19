import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate


cache = {}

sns.set(
    context="paper",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k", "figure.figsize": (4, 3)},
)


def generate_dataset() -> (np.ndarray, np.ndarray):
    """Returns a classification dataset with fixed parameters"""
    X, y = datasets.make_classification(
        n_samples=20,
        n_features=2,
        n_redundant=0,
        class_sep=1.0,
        n_clusters_per_class=1,
        random_state=0,
    )
    return X, y


def plot_two_class_dataset(X_test: np.ndarray, X_train: np.ndarray, y_test: np.ndarray, y_train: np.ndarray) -> None:
    """Plots a dataset with two classes which has been split into training an test set"""
    plt.scatter(*X_train[y_train == 0].T, marker='x', color='lightblue', label="Train Set, Class 0")
    plt.scatter(*X_train[y_train == 1].T, marker='o', color='lightblue', label="Train Set, Class 1")
    plt.scatter(*X_test[y_test == 0].T, marker='x', color='orange', label="Test Set, Class 0")
    plt.scatter(*X_test[y_test == 1].T, marker='o', color='orange', label="Test Set, Class 1")

    train = mlines.Line2D([], [], color='lightblue', marker='.', linestyle='None', markersize=6, label='Train')
    test = mlines.Line2D([], [], color='orange', marker='.', linestyle='None', markersize=6, label='Test')
    class_0 = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=6, label='Class 0')
    class_1 = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=6, label='Class 1')

    plt.legend(handles=[train, test, class_0, class_1], loc="upper right")

    # Create the figure
    plt.xlabel("$X_0$")
    plt.ylabel("$X_1$")
    plt.title("Train/Test Split")
    plt.show()


def measure_accuracy(clf: sklearn.base.BaseEstimator, X_test: np.ndarray, X_train: np.ndarray,
                     y_test: np.ndarray, y_train: np.ndarray) -> (float, float):
    """Measures the accuracy of an sklearn estimator on the given dataset"""
    clf.fit(X_train, y_train)
    # Predict
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    # Eval
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    return acc_test, acc_train


def plot_decision_boundary(model: KNeighborsClassifier, X: np.ndarray, y: np.ndarray) -> None:
    """Create a decision boundary plot that shows the predicted label for each point."""
    h = 0.05  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    k = model.n_neighbors
    if k in cache:
        Z = cache[k]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        cache[k] = Z

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)
    plt.title("2D Iris KNN Decision Boundaries ($k=%s$)" % k)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def plot_iris_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """Plots the iris dataset as a 2d scatter plot"""
    # Create color maps
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    # Plot data
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors="k")
    plt.title("2D Iris Dataset")
    plt.show()


def evaluate(k: int, X: np.ndarray, y: np.ndarray) -> (float, float):
    """Runs 10 fold cross-validation of a KNN model and returns the mean training accuracy and mean test accuracy."""
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_validate(
        estimator=model, X=X, y=y, scoring="accuracy", cv=10, return_train_score=True
    )
    return np.mean(scores["train_score"]), np.mean(scores["test_score"])


def plot_polyfit(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray,
                 degree: int, test_error: float) -> None:
    """Plots a fitted polynomial regression curve."""
    plt.title(f"d = {degree}, test-mse = {test_error:2.3f}")
    plt.plot(x_test, y_pred, c="r", label="polynomial fit")
    plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.scatter(x_train, y_train, label="training data")
    plt.ylim(-1.5, 1.5)
    plt.xlabel("x")
    plt.ylabel("y")


def plot_sin_curve(x_test: np.ndarray, x_train: np.ndarray, y_test: np.ndarray, y_train: np.ndarray) -> None:
    """Plots a sinus curve."""
    plt.figure()
    plt.scatter(x_train, y_train, label="training data")
    plt.plot(x_test, y_test, c="g", label="$\sin(2 \pi x)$")
    plt.legend()
    plt.show()


def fsin(x: np.ndarray) -> np.ndarray:
    """Map to sinus function."""
    return np.sin(2 * np.pi * x)
