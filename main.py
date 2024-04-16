from niapy.algorithms.basic import GeneticAlgorithm
from niapy.task import Task
from Clustering import Clustering
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, homogeneity_score

datasets_list = [
    ("Iris", datasets.load_iris().data[:, :2], datasets.load_iris().target),
    ("Wine", datasets.load_wine().data[:, :2], datasets.load_wine().target),
    ("Noisy Circles", *datasets.make_circles(n_samples=100, factor=0.5, noise=0.05)),
    ("Noisy Moons", *datasets.make_moons(n_samples=100, noise=0.05)),
]

num_clusters = 3
num_features = 2
population_size = 70
crossover_probability = 0.8
mutation_probability = 0.1
max_evals = 1000

algorithm = GeneticAlgorithm(population_size=population_size, crossover_probability=crossover_probability, mutation_probability=mutation_probability)

for dataset_name, dataset_data, dataset_labels in datasets_list:
    np.random.seed(30)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(dataset_data)

    clustering_problem = Clustering(num_clusters=num_clusters, num_features=num_features,
                                     dimension=num_features * num_clusters, instances=normalized_data, lower=0, upper=1)

    task = Task(problem=clustering_problem, max_evals=max_evals)

    best_solution, best_solution_fitness = algorithm.run(task)

    centroids = best_solution.reshape(num_clusters, num_features)

    silhouette = silhouette_score(normalized_data, clustering_problem.cluster_indices)
    homogeneity = homogeneity_score(dataset_labels, clustering_problem.cluster_indices)

    print("Dataset:", dataset_name)
    print("Best solution found:", best_solution)
    print("Best solution fitness:", best_solution_fitness)
    print("Silhouette score:", silhouette)
    print("Homogeneity score:", homogeneity)
    print("\n")
