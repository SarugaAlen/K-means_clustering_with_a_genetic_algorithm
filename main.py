
from niapy.algorithms.basic import GeneticAlgorithm
from sklearn.cluster import KMeans
from niapy.task import Task
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from Clustering import Clustering
from kmeans_clustering import perform_kmeans_clustering
import matplotlib.pyplot as plt


def visualize_clusters(data, labels, centroids, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=150, marker='X', c='red', label='Optimized Centroids')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


iris_dataset = load_iris()
"""Tukaj uporabim samo 2 feature-a, ker je lažje vizualizirati rezultate clusteringa."""
# perform_kmeans_clustering(iris_data)

iris_data = iris_dataset.data[:, :2]
iris_labels = iris_dataset.target
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(iris_data)

for i in range(10):

    num_clusters = 2
    num_features = 2

    clustering_problem = Clustering(num_clusters=num_clusters, dimension=num_features, instances=normalized_data, lower=0,
                                    upper=1)

    task = Task(problem=clustering_problem, max_evals=100)

    algorithm = GeneticAlgorithm(population_size=50, crossover_probability=0.9, mutation_probability=0.2)

    best = algorithm.run(task)

    print("Best solution found: ", best)

    optimized_centroids = best[0]
    optimized_centroids = optimized_centroids.reshape(-1, 2)
    print("Shape of optimized centroids:", optimized_centroids.shape)

    # Visualize the clusters with the optimized centroids
    visualize_clusters(normalized_data, iris_labels, optimized_centroids, "Clusters with Optimized Centroids")

