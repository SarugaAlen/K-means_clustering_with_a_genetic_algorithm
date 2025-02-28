import numpy as np
from niapy.algorithms.basic import GeneticAlgorithm
from niapy.task import Task
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.cluster import KMeans
from Clustering import Clustering
from visualization import visualize_clusters
import timeit

datasets_list = [
    ("Iris", datasets.load_iris().data[:, :], datasets.load_iris().target, 3),
    ("Wine", datasets.load_wine().data[:, :], datasets.load_wine().target, 3),
    ("Noisy Circles", *datasets.make_circles(n_samples=100, factor=0.5, noise=0.05), 3),
    ("Noisy Moons", *datasets.make_moons(n_samples=100, noise=0.05), 5),
]

population_size = 1000
crossover_probability = 0.7
mutation_probability = 0.5

max_evals = 500_000

algorithm = GeneticAlgorithm(population_size=population_size, crossover_probability=crossover_probability,
                             mutation_probability=mutation_probability)

for dataset_name, dataset_data, dataset_labels, num_clusters in datasets_list:
    normalized_data = MinMaxScaler().fit_transform(dataset_data)
    num_features = dataset_data.shape[1]

    # KMeans
    kmeans_start_time = timeit.default_timer()
    kmeans = KMeans(n_clusters=num_clusters, init='random', n_init='auto', max_iter=300)
    kmeans.fit(normalized_data)
    silhouette_kmeans = silhouette_score(normalized_data, kmeans.labels_)
    homogeneity_kmeans = homogeneity_score(dataset_labels, kmeans.labels_)
    visualize_clusters(normalized_data, kmeans.labels_, kmeans.cluster_centers_, num_clusters,
                       f'KMeans Clustering ({dataset_name})')
    kmeans_end_time = timeit.default_timer()
    kmeans_time = kmeans_end_time - kmeans_start_time

    # Nia
    nia_start_time = timeit.default_timer()
    clustering_problem = Clustering(num_clusters=num_clusters, num_features=num_features,
                                    dimension=num_features * num_clusters, instances=normalized_data, lower=0, upper=1)
    task = Task(problem=clustering_problem, max_evals=max_evals)
    best_solution, best_solution_fitness = algorithm.run(task)
    centroids = best_solution.reshape(num_clusters, num_features)
    silhouette_nia = silhouette_score(normalized_data, clustering_problem.cluster_indices)
    homogeneity_nia = homogeneity_score(dataset_labels, clustering_problem.cluster_indices)
    visualize_clusters(normalized_data, clustering_problem.cluster_indices, centroids, num_clusters,
                       f'Nia Clustering ({dataset_name})')
    nia_end_time = timeit.default_timer()
    nia_time = nia_end_time - nia_start_time

    print("Dataset:", dataset_name)
    print("KMeans Execution time:", kmeans_time, "seconds")
    print("Silhouette score (KMeans):", silhouette_kmeans)
    print("Homogeneity score (KMeans):", homogeneity_kmeans)
    print("\n")
    print("Nia Execution time:", nia_time, "seconds")
    print("Best solution found:", best_solution)
    print("Best solution fitness:", best_solution_fitness)
    print("Silhouette score (Nia):", silhouette_nia)
    print("Homogeneity score (Nia):", homogeneity_nia)
    print("\n")
