#!/usr/bin/env python
# coding: utf-8
# Referenzen
# - https://scikit-learn.org/stable/modules/ensemble.html#forest
# - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn import datasets
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
)


def cross_val_dt_rt(load_fun) -> (dict, dict):
    """Conducts a 10-fold cross validation for a decision tree and random forrest and
     returns their accuracy scores given an sklearn discrete dataset loader function."""
    raise NotImplementedError


def cross_val_dt_rf_continuous(load_fun) -> (dict, dict):
    """Conducts a 10-fold cross validation for a decision tree and random forrest and
     returns their mse scores given an sklearn discrete dataset loader function."""
    raise NotImplementedError


def main():
    # Get the data loader
    loader_classification = [
        ("Iris", datasets.load_iris),
        ("Digits", datasets.load_digits),
        ("Wine", datasets.load_wine),
        ("Breast Cancer", datasets.load_breast_cancer),
    ]
    loader_regression = [
        ("Boston", datasets.load_boston),
        ("Diabetes", datasets.load_diabetes),
        ("Linnerud", datasets.load_linnerud),
    ]

    # Set seed for reproducibility
    np.random.seed(0)

    # For each classification dataset evaluate:
    for name, load_fun in loader_classification:
        scores_dt, scores_rf = cross_val_dt_rt(load_fun)

        # Print results
        print(f"-- Dataset: [{name}]")
        print(f"              \t\tAccuracy (%)")
        print(f"              \tTrain     \tTest")
        print(
            f"Decision Tree\t{scores_dt['train_score'].mean()*100:.2f}\t\t{scores_dt['test_score'].mean()*100:.2f}"
        )
        print(
            f"Random Forest\t{scores_rf['train_score'].mean()*100:.2f}\t\t{scores_rf['test_score'].mean()*100:.2f}\n\n"
        )

    # For each regression dataset evaluate:
    for name, load_fun in loader_regression:
        scores_dt, scores_rf = cross_val_dt_rf_continuous(load_fun)

        # Print results
        print(f"-- Dataset: [{name}]")
        print(f"              \t\tRMSE")
        print(f"              \tTrain     \tTest")
        print(
            f"Decision Tree\t{np.sqrt(scores_dt['train_score'].mean()):.2f}\t\t{np.sqrt(scores_dt['test_score'].mean()):.2f}"
        )
        print(
            f"Random Forest\t{np.sqrt(scores_rf['train_score'].mean()):.2f}\t\t{np.sqrt(scores_rf['test_score'].mean()):.2f}\n\n"
        )


if __name__ == '__main__':
    main()
