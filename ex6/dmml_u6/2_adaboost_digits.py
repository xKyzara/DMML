# coding: utf-8
# Referenzen
# - https://scikit-learn.org/stable/modules/ensemble.html#ada_boost
# [2] T. Hastie, R. Tibshirani and J. Friedman, The Elements of Statistical Learning, Second Edition, Section 10.13.2, Springer, 2009.

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import utils


def plot_digit_dataset(X_train: np.ndarray, y_train: np.ndarray) -> None:
    """Plots the first 5 images of the digit dataset."""
    plt.figure()
    for i, (x_i, y_i) in enumerate(zip(X_train[:5], y_train[:5]), start=1):
        plt.subplot(150 + i)
        plt.imshow(x_i.reshape(8, 8), cmap="gray")
        plt.title("label = " + str(y_i))
        plt.axis("off")
    plt.show()


def get_acc_for_estimators(base_estimator: DecisionTreeClassifier, X_train: np.ndarray,
                           X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                           n_estimators: int, learning_rate: float):
    """Fits an AdaBoostClassifier on the given training data and returns the training and test accuracy for all
     iterations."""
    # Create AdaBoost model
    model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate)
    # Fit model
    model.fit(X_train, y_train)
    # Evaluate model on train/test set
    accuracy_train = [
        accuracy_score(y_pred, y_train) for y_pred in model.staged_predict(X_train)
    ]
    accuracy_test = [
        accuracy_score(y_pred, y_test) for y_pred in model.staged_predict(X_test)
    ]
    return model, accuracy_train, accuracy_test


def main():
    # Load the digits dataset
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print("Digits shape:", X.shape)

    # Plot samples
    plot_digit_dataset(X_train, y_train)

    # Create a Decision Tree with depth of 1 (stump) as weak base learner and a strong base learner with depth 10
    stump = DecisionTreeClassifier(max_depth=1)
    model, accuracy_train, accuracy_test = get_acc_for_estimators(stump, X_train, X_test, y_train, y_test,
                                                                  n_estimators=300, learning_rate=0.01)
    # Plot
    utils.plot_estimator_performance(accuracy_train, accuracy_test)

    # Plot
    plt.figure()
    # Run with specific learning rate
    for lr in [2, 1, 1 / 2, 1 / 4, 1 / 8]:
        _, accuracy_train, _ = get_acc_for_estimators(stump, X_train, X_test, y_train, y_test,
                                                                      n_estimators=300, learning_rate=lr)
        plt.plot(accuracy_train, label=f"$lr={lr}$")
    plt.legend()
    plt.suptitle("AdaBoost Learning Rate Comparison")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Training Accuracy")
    plt.show()

    utils.cross_val_lr(X, y)


if __name__ == '__main__':
    main()
