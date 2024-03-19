import numpy as np
from niapy.problems import Problem
from sklearn.metrics import pairwise_distances_argmin_min

class Clustering(Problem):
    def __init__(self, cluster_centers, dimension, lower=0, upper=1, instances=None, *args, **kwargs):
        super().__init__(dimension=dimension, lower=lower, upper=upper, *args, **kwargs)
        self.cluster_centers = cluster_centers
        self.instances = instances

    def _evaluate(self, x, instances=None):
        if instances is None:
            instances = self.instances

        centroids = x.reshape((self.cluster_centers, -1))

        distances = self._calculate_distances(centroids, instances)
        cluster_assignments = np.argmin(distances, axis=1)

        # Calculate fitness based on some clustering evaluation metric (e.g., silhouette score, inertia, etc.)
        fitness = self._calculate_fitness(distances, cluster_assignments)

        return fitness

    # 1 korak iz x dobis lokacijen za vse centroide
    # 2 korak za vsako instanco izracunas razdaljo do vseh centroidov
    # 3 korak dodeli instance najblizjemu centroidu
    # 4 korak izracunas kakovost
    # minimization manjsi fitnes je bolsi
    def _calculate_distances(self, centroids, instances):
        """
        Calculate the Euclidean distance between each instance and all cluster centers.

        Parameters:
        - centroids: 2D array, representing cluster centers
        - instances: 2D array, representing data instances

        Returns:
        - distances: 2D array, where each row corresponds to distances from an instance to all cluster centers
        """
        distances = np.linalg.norm(instances[:, np.newaxis, :] - centroids, axis=2)
        return distances


    def _calculate_fitness(self, distances, cluster_assignments):
        """
        Calculate fitness based on clustering evaluation metric.

        Parameters:
        - distances: 2D array, distances from each instance to all cluster centers
        - cluster_assignments: 1D array, cluster assignments for each instance

        Returns:
        - fitness: Fitness value representing clustering quality
        """
        # Example: You can use inertia as a fitness measure
        inertia = np.sum(np.min(distances, axis=1))
        return inertia
