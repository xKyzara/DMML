import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
)

cache = {}


def plot_decision_boundary(model, X, y):
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

    k = model.min_samples_leaf
    if k in cache:
        Z = cache[k]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        cache[k] = Z

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_decision_boundary_adaboost(classifier, X, y, N=1000, ax=None):
    """Utility function to plot decision boundary and scatter plot of data"""
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))

    # Initialize the sample weights
    sample_weight = np.ones(X.shape[0]) / X.shape[0]

    for i, zz in enumerate(
        classifier.staged_predict_proba(np.c_[xx.ravel(), yy.ravel()])
    ):
        plt.subplot(classifier.n_estimators, 2, 2 * i + 2)
        plt.title(f"Ensemble at Step {i + 1}")
        Z = zz[:, 1].reshape(xx.shape)
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        # Get current axis and plot
        ax = plt.gca()
        ax.contourf(xx, yy, Z, 2, cmap="RdBu", alpha=0.5)
        ax.contour(xx, yy, Z, 2, cmap="RdBu")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
        ax.set_xlabel("$X_0$")
        ax.set_ylabel("$X_1$")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(classifier.n_estimators, 2, 2 * i + 1)
        plt.title(f"Decision Stump at Step {i + 1}")
        zz = classifier.estimators_[i].predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = zz[:, 1].reshape(xx.shape)

        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        # Get current axis and plot
        ax = plt.gca()
        ax.contourf(xx, yy, Z, 2, cmap="RdBu", alpha=0.5)
        ax.contour(xx, yy, Z, 2, cmap="RdBu")
        s_weights = (sample_weight / sample_weight.sum()) * 4000
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, s=s_weights)
        ax.set_xlabel("$X_0$")
        ax.set_ylabel("$X_1$")
        plt.xticks([])
        plt.yticks([])

        # Update samples weights
        y_pred = classifier.estimators_[i].predict(X)
        incorrect = y_pred != y
        estimator_weight = classifier.estimator_weights_[i]
        sample_weight *= np.exp(
            estimator_weight
            * incorrect
            * ((sample_weight > 0) | (estimator_weight < 0))
        )

    plt.tight_layout()
    plt.suptitle(
        "AdaBoost Decision Boundaries for\nDecision Stumps (left, scale indicates sample weight)\nand Ensemble (right) after each Iteration",
        x=0.5,
        y=1.05,
    )

# Run 10 fold cross-validation of the AdaBoost model
def evaluate(X: np.ndarray, y, learning_rate: np.ndarray):
    model = AdaBoostClassifier(n_estimators=300, learning_rate=learning_rate)
    scores = cross_validate(
        estimator=model,
        X=X,
        y=y,
        scoring="accuracy",
        cv=10,
        return_train_score=True,
        n_jobs=2,
    )
    return np.mean(scores["train_score"]), np.mean(scores["test_score"])


def plot_estimator_performance(accuracy_train, accuracy_test):
    plt.figure()
    plt.plot(accuracy_train, label="Train")
    plt.plot(accuracy_test, label="Test")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.title("AdaBoost Accuracy over Iterations")
    plt.legend()
    plt.show()


def plot_lr_comparision_cross_val(acc_train, acc_test, learning_rates):
    plt.figure()
    plt.scatter(learning_rates, acc_train, label="Accuracy Train", marker="x")
    plt.scatter(learning_rates, acc_test, label="Accuracy Test", marker="^")
    plt.legend()
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.xscale("log", basex=2)
    plt.title("AdaBoost: Accuracy vs. Learning Rate")
    plt.show()


def cross_val_lr(X, y):
    # Define validation interval
    learning_rates = [2 ** i for i in range(-5, 4)]
    # Evaluate each learning rate
    accuracies = np.array([evaluate(X, y, lr) for lr in tqdm(learning_rates)])
    acc_train = accuracies[:, 0]
    acc_test = accuracies[:, 1]
    # Plot results
    plot_lr_comparision_cross_val(acc_train, acc_test, learning_rates)


def visualize_gradient_boosting_sine(X, y, axs, staged_y_preds, i, step):
    # Plot sine curve fitting
    ax = axs[i//step]
    ax.set_xlim(0, X.max())
    ax.set_ylim(-1, 1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$sin(x) / Prediction$")

    # Obtain lines
    error_lines = []
    for _ in range(X.shape[0]):
        (l,) = ax.plot([], [], c="red", alpha=0.4)
        error_lines.append(l)

    (line_true,) = ax.plot(X, y, lw=2)
    (line_pred,) = ax.plot([], [], lw=2)

    # Enable legend
    ax.legend(
        [line_true, line_pred, error_lines[0]],
        ["True", "Predicted", "Residuals"],
        loc="upper right",
    )

    # Animation function
    print(f"Progress: {i/len(staged_y_preds) * 100:.1f}%", end="\r")

    # Set error bars
    for j, err_l in enumerate(error_lines):
        err_l.set_data(
            [X[j], X[j]], [y[j], staged_y_preds[i][j]]
        )

    # Set prediction
    line_pred.set_data(X[:, 0], staged_y_preds[i])

    mse = mean_squared_error(y, staged_y_preds[i])
    ax.set_title(f"Iteration: {i}, MSE: {mse:.5f}", y=0.05)


def plot_sine(X, y):
    plt.figure()
    plt.plot(X, y)
    plt.xlabel("$x$")
    plt.ylabel("$sin(x)$")
    plt.title("Sinus Data")
    plt.show()


def plot_loss(errors):
    plt.figure()
    plt.plot(errors)
    plt.xlabel("Iteration")
    plt.ylabel("Least Squares Loss")
    plt.title("Gradient Boosting: Loss over Iterations")
    plt.show()


def annotate_axes_lr(ax):
    ax[0].set_title("Train")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("MSE")
    ax[1].set_title("Test")
    ax[1].set_xlabel("Iteration")
    ax[0].set_ylim(-1, 15)
    ax[1].set_ylim(10, 40)
    plt.savefig("gradient_boosting_lr.pdf")
    plt.legend()
    plt.show()
