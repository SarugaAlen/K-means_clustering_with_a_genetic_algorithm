import numpy as np
from niapy.problems import Problem

class Clustering(Problem):
    def __init__(self, num_clusters, dimension, lower=-10, upper=10, instances=None, *args, **kwargs):
        super().__init__(dimension=dimension, lower=lower, upper=upper, *args, **kwargs)
        self.num_clusters = num_clusters
        self.instances = instances

    def euclidean_distance(self, Y):
        # Calculate the Euclidean distance between each instance and all cluster centers
        return np.sqrt(np.sum(np.power(self.instances - Y, 2), axis=1))

    def _evaluate(self, x):
        clusters = []
        clusters_sum_dist = []
        all_clusters_dists = []

        cluster_centers = x

        # Step 2: Compute distances from instances to all cluster centers
        for instance in self.instances:
            # Calculate distances from each instance to all cluster centers
            instance_dists = [self.euclidean_distance(cluster_center) for cluster_center in cluster_centers]
            all_clusters_dists.append(np.array(instance_dists))

        all_clusters_dists = np.array(all_clusters_dists)

        # Step 3: Assign instances to the nearest cluster center
        cluster_indices = np.argmin(all_clusters_dists, axis=1)

        for clust_idx in range(self.num_clusters):
            cluster_instances = np.where(cluster_indices == clust_idx)[0]
            clusters.append(cluster_instances)

            if len(cluster_instances) == 0:
                # If a cluster has 0 samples, set its total distance to 0
                clusters_sum_dist.append(0)
            else:
                # Calculate the sum of distances in each cluster
                clusters_sum_dist.append(np.sum(all_clusters_dists[cluster_instances, clust_idx]))

        clusters_sum_dist = np.array(clusters_sum_dist)

        # Step 5: Sum the distances in all clusters
        total_distance = np.sum(clusters_sum_dist)

        # Step 6: Calculate the inverse of the sum of distances (fitness value)
        fitness = 1.0 / (total_distance + 0.00000001)

        return fitness
