from niapy.algorithms.basic import GeneticAlgorithm
from niapy.task import Task
from Clustering import Clustering
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def visualize_clusters(data, labels, centroids, num_clusters, title):
    plt.figure(figsize=(8, 6))

    # Define color map based on the number of clusters
    cmap = plt.cm.get_cmap('viridis', num_clusters)

    # Plot data points for each pair of features
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            plt.scatter(data[:, i], data[:, j], c=labels, cmap=cmap, label=f'Feature {i + 1} vs Feature {j + 1}')

    # Plot cluster centers with circles colored according to the cluster they belong to
    for idx, centroid in enumerate(centroids):
        cluster_color = cmap(idx % num_clusters)  # Use modulo to ensure cyclic selection of colors
        plt.scatter(centroid[0], centroid[1], s=100, marker='o', c=[cluster_color], edgecolor='black',
                    label=f'Cluster Center {idx + 1}')

    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Feature')
    #plt.legend()
    plt.show()

iris_dataset = load_iris()
iris_data = iris_dataset.data[:, :2]
iris_labels = iris_dataset.target

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(iris_data)

num_clusters = 2
num_features = 2

for i in range(10):
    # Define the clustering problem
    clustering_problem = Clustering(num_clusters=num_clusters, dimension=num_features*num_clusters, instances=normalized_data, lower=0, upper=1)

    # Define the optimization task
    task = Task(problem=clustering_problem, max_evals=1000)

    # Initialize and run the genetic algorithm
    algorithm = GeneticAlgorithm(population_size=100, crossover_probability=0.8, mutation_probability=0.1)
    best = algorithm.run(task)

    print(best)
    # Extract the centroids
    centroids = best[0].reshape(num_clusters, num_features)

    print("Best solution found:", best)
    print("Extracted centroids:", centroids)

    # Visualize the clusters with the optimized centroids
    visualize_clusters(normalized_data, iris_labels, centroids, num_clusters, "Clusters with Optimized Centroids")
