
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
# perform_kmeans_clustering(iris_data)

for i in range(1):
    iris_data = iris_dataset.data[:, :2]
    my_instances = iris_data
    num_clusters = 2
    num_features = 2

    clustering_problem = Clustering(num_clusters=num_clusters, dimension=num_features, instances=my_instances, lower=0,
                                    upper=1)

    task = Task(problem=clustering_problem, max_evals=100)

    algorithm = GeneticAlgorithm(population_size=50, crossover_probability=0.9, mutation_probability=0.2)

    best = algorithm.run(task)

    print("Best solution found: ", best)



"""
clusters = []
cluster_centers = []
clusters_sum_dist = []
all_clusters_dists = []

random_indices = np.random.choice(iris_data.shape[0], size=num_clusters, replace=False)
cluster_centers = np.array(iris_data[random_indices])

print("Initial Cluster Centers:")
print(cluster_centers)
'''' PRVI DEL OKEJ '''

''' DRUGI DEL izracun razdalje do centroidov '''
for clust_idx in range(num_clusters):
    cluster_center_dists = euclidean_distance(my_instances, cluster_centers[clust_idx ])
    all_clusters_dists.append(np.array(cluster_center_dists))

all_clusters_dists = np.array(all_clusters_dists)
print("Distances to Cluster Centers:")
print(all_clusters_dists)

''' TRETJI del dodeli instance najblizjemu centroidu '''
cluster_indices = np.argmin(all_clusters_dists, axis=0)
print("Cluster Indices:")
print(cluster_indices)

for clust_idx in range(num_clusters):
    cluster_instances = np.where(cluster_indices == clust_idx)[0]
    clusters.append(cluster_instances)

    if len(cluster_instances) == 0:
        clusters_sum_dist.append(0)
    else:
        clusters_sum_dist.append(np.sum(all_clusters_dists[clust_idx, cluster_instances]))

clusters_sum_dist = np.array(clusters_sum_dist)

print("Clusters:")
print(clusters)
print("Clusters Sum Distances:")
print(clusters_sum_dist)

''' CETRTI DEL izracun kakovosti '''

fitness = 1.0 / np.sum(clusters_sum_dist + 0.00000001)
print(fitness)
"""