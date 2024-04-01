from niapy.algorithms.basic import GeneticAlgorithm
from niapy.task import Task
from Clustering import Clustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from visualization import visualize_clusters
from kmeans_clustering import perform_kmeans_clustering

iris_dataset = load_iris()
iris_data = iris_dataset.data[:, :2]
iris_labels = iris_dataset.target

perform_kmeans_clustering(iris_data, n_clusters=3)

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(iris_data)

num_clusters = 3
num_features = 2

clustering_problem = Clustering(num_clusters=num_clusters, num_features=num_features,
                                dimension=num_features * num_clusters, instances=normalized_data, lower=0, upper=1)

task = Task(problem=clustering_problem, max_evals=1000)

algorithm = GeneticAlgorithm(population_size=70, crossover_probability=0.8, mutation_probability=0.1)
best = algorithm.run(task)

centroids = best[0].reshape(num_clusters, num_features)

print("Best solution found:", best)

visualize_clusters(normalized_data, iris_labels, centroids, num_clusters, "Gručenje z optimiziranimi centroidi")
