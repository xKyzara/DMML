#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
)


def plot_pr_curve(y: np.ndarray, y_scores: np.ndarray):
    """Plots the precision recall curve given the true class labels y and the predicted labels with their probability"""
    precision, recall, thresholds = precision_recall_curve(y, y_scores[:, 1])
    plt.plot(recall, precision, marker='.', label='Random Forest')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(linestyle="--")
    plt.legend()
    plt.show()


def plot_roc_curve(y, y_scores):
    """Plots the Receiver Operating Characteristic Curve given the true class labels y and the predicted labels with
     their probability"""
    fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_scores[:, 1])
    roc_auc = roc_auc_score(y_true=y, y_score=y_scores[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = "Receiver operating characteristic curve (AUC score"
    plt.show()


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray):
    """Plots the confusion matrix given the true test labels y_test and the predicted labes y_pred"""
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    sns.heatmap(confmat, annot=True, fmt="d", vmin=0, vmax=10)
    plt.xlabel("Predicted Number")
    plt.ylabel("True Number")
    plt.show()


def main():
    # Load Data
    X, y = datasets.load_breast_cancer(return_X_y=True)
    clf = RandomForestClassifier()

    # Compute cross validation probabilities for each sample
    y_scores = cross_val_predict(
        estimator=clf, X=X, y=y, cv=5, n_jobs=-1, verbose=0, method="predict_proba"
    )

    plot_pr_curve(y, y_scores)
    plot_roc_curve(y, y_scores)

    # Load digits dataset
    X, y = datasets.load_digits(return_X_y=True)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Train classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plot_confusion_matrix(y_test, y_pred)


if __name__ == '__main__':
    main()
