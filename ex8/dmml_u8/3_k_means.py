#!/usr/bin/env python
# coding: utf-8
# ## Referenzen
# https://scikit-learn.org/stable/modules/clustering.html#k-means
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles, make_moons

sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 100, "scatter.edgecolors": "k"},
)


def plot_cluster_data(X):
    """Creates a scatter plot for the given data."""
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.axis("off")
    plt.show()


def fit_kmeans(X, k, init="k-means++", n_init=10, random_state=0) -> [np.ndarray, np.ndarray]:
    """Runs K-Means clustering and returns the label assignments and cluster centroids."""
    # Build K-Means model
    model = KMeans(n_clusters=k, n_init=n_init, init=init, random_state=random_state)
    # Fit model
    model.fit(X)
    # Predict cluster assignment for each datapoint
    labels = model.predict(X)
    # Get cluster centers
    centers = model.cluster_centers_
    return labels, centers


def plot_clustering(X, labels, centers=None, title="", subplot=None):
    """Helper function to plot the clustering"""
    # Plot in given subplot
    if subplot:
        plt.subplot(subplot)

    # Plot data with labels as color
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    # Plot centers if given
    if centers is not None:
        plt.scatter(
            centers[:, 0], centers[:, 1], c = "red", s = 150, alpha = 0.9, label = "Centers"
        )
        # Set title
        plt.title(title)
        plt.axis("off")


def plot_kmeans_iterations(X, guess_center, initial_labels):
    """Plots the cluster assignments after iteration 1..6."""
    plt.figure(figsize=(2 * 3, 3 * 3))
    plot_clustering(
        X, initial_labels, centers = guess_center, title = "Iteration 0", subplot = 321
    )
    for i in range(1, 6):
        # Stop K-Means after i iterations
        model = KMeans(n_clusters=3, init=guess_center, n_init=1, max_iter=i).fit(X)
        labels = model.predict(X)
        centers = model.cluster_centers_
        # Plot clustering for current iteration
        plot_clustering(X, labels, centers, title="Iteration " + str(i), subplot=321 + i)
    plt.show()


def plot_kmeans(X, k, title, subplot=None):
    """# Helper function to run K-Means with given k and plot clustering"""
    # Run K-Means
    labels, centers = fit_kmeans(X=X, k=k)

    # Plot resulting clustering
    plot_clustering(X, labels, centers, title, subplot)


def main():
    part_1()
    part_2()
    part_3()


def part_1():
    # Generate three clusters
    X, y = make_blobs(
        n_samples=1000, centers=3, cluster_std=0.6, n_features=2, random_state=0
    )

    # Plot data
    plot_cluster_data(X)
    labels, centers = fit_kmeans(X, k=3)

    # Plot clustering
    plt.figure()
    plot_clustering(X, labels, centers)
    plt.show()

    # Generate 3 clusters
    true_centers = [[0, 0], [1, 0], [0, 1]]
    X, y = make_blobs(
        n_samples=900, centers=true_centers, cluster_std=0.25, n_features=2, random_state=0
    )

    # Generate random centers
    np.random.seed(2)
    guess_center = np.array(true_centers) + np.random.randn(3, 2) * 0.5

    # For each datapoint find closest center
    initial_labels = np.argmin(
        np.sqrt(((X - guess_center.reshape(3, 1, 2)) ** 2).sum(2)), 0
    )

    plot_kmeans_iterations(X, guess_center, initial_labels)


def part_2():
    # Generate four clusters with custom centers
    X, y = make_blobs(
        n_samples=[100, 100, 100, 500],
        centers=[[0, 0], [0, 0.25], [0.15, 0.15], [0.5, 1]],
        cluster_std=[0.04, 0.04, 0.04, 0.1],
        n_features=2,
        random_state=0,
    )

    # Run kmeans twice with two different random states
    labels_a, centers_a = fit_kmeans(X=X, k=4, n_init=1, init="random", random_state=0)
    labels_b, centers_b = fit_kmeans(X=X, k=4, n_init=1, init="random", random_state=1)

    # Plot cluster assignments and cluster centers
    plt.figure(figsize=(10, 5))
    plot_clustering(X, labels_a, centers_a, title="Bad Clustering", subplot=121)
    plot_clustering(X, labels_b, centers_b, title="Good Clustering", subplot=122)
    plt.show()

    # Generate three clusters
    X, y = make_blobs(
        n_samples=1000, centers=3, cluster_std=0.6, n_features=2, random_state=0
    )

    # Run and plot
    plt.figure(figsize=(15, 5))
    plot_kmeans(X, k=2, title="$K=2$, Too Few", subplot=131)
    plot_kmeans(X, k=3, title="$K=3$, Correct", subplot=132)
    plot_kmeans(X, k=4, title="$K=4$, Too Many", subplot=133)
    plt.show()


def part_3():
    # Make data
    X_circles, _ = make_circles(n_samples=1000, factor=0.5, noise=0.05)
    X_moons, _ = make_moons(1000, noise=0.03, random_state=0)

    # Run K-Means and Plot
    plt.figure(figsize=(10, 5))
    plot_kmeans(X_circles, k=2, title="Circles", subplot=121)
    plot_kmeans(X_moons, k=2, title="Moons", subplot=122)
    plt.show()

    # Make clusters with different variances
    X, y = make_blobs(
        n_samples=2000, centers=[[0.5, 0.0], [0.5, 5.5]], cluster_std=[0.15, 2.0]
    )

    # Plot true and K-Means clustering
    plt.figure(figsize=(10, 5))
    plot_clustering(X, labels=y, title="True Clustering", subplot=121)
    plot_kmeans(k=2, X=X, title="K-Means Clustering", subplot=122)
    plt.show()

    # Make data
    X = np.random.rand(5000, 2)

    # Run and plot K-Means
    plt.figure(figsize=(10, 5))
    plot_clustering(X, labels=np.zeros(X.shape[0]), title="Uniform Data", subplot=121)
    plot_kmeans(k=10, X=X, title="K-Means Clustering", subplot=122)
    plt.show()


if __name__ == '__main__':
    main()
