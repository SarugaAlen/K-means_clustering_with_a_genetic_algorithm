from niapy.algorithms.basic import GeneticAlgorithm
from sklearn.cluster import KMeans
from niapy.task import Task
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin_min
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
# iris_data = iris_dataset.data[:, :2]
# perform_kmeans_clustering(iris_data)

a = 100
for i in range(1):
    iris_data = iris_dataset.data[:, :2]
    iris_target = iris_dataset.target
    print("To je data", iris_data)
    print(".................................")
    print("to je target", iris_target)

    my_instances = iris_data
    num_clusters = 2
    num_features = 2
    clustering_problem = Clustering(num_clusters=num_clusters, dimension=num_features, instances=my_instances, lower=0,
                                    upper=1)
    # clustering_task = Task(problem=clustering_problem, max_evals=100)

    # algorithm = GeneticAlgorithm(population_size=50, crossover_probability=0.9, mutation_probability=0.2)

    x = np.array([5.5, 8, 8, 3.5])

    cluster_centers = x.reshape((num_clusters, num_features))
    print("Cluster Centers:")
    print(cluster_centers)

    '''' PRVI DEL OKEJ '''

    # best_solution, _ = algorithm.run(clustering_task)

# print(f"Best Solution - {best_solution}")

# Perform clustering with the optimized solution
# optimized_centroids = np.array(best_solution).reshape((num_clusters, num_features))
# labels_optimized = pairwise_distances_argmin_min(iris_data, optimized_centroids)[0]

# Visualize the clustering results after optimization
# visualize_clusters(iris_data, labels_optimized, optimized_centroids, 'Genetic Algorithm Optimized Clustering')

# clustering_quality = clustering_problem._evaluate(best_solution)
# print("Clustering Quality:", clustering_quality)
