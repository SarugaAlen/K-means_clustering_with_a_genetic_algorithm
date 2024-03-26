import numpy as np
from niapy.problems import Problem
from sklearn.metrics import pairwise_distances_argmin_min


class Clustering(Problem):
    def __init__(self, num_clusters, dimension, lower=0, upper=1, instances=None, *args, **kwargs):
        super().__init__(dimension=dimension, lower=lower, upper=upper, *args, **kwargs)
        self.num_clusters = num_clusters
        self.instances = instances

    def euclidean_distance(X, Y):
        return np.sqrt(np.sum(np.power(X - Y, 2), axis=1))

    # 1 korak iz x dobis lokacijen za vse centroide

    # 2 korak za vsako instanco izracunas razdaljo do vseh centroidov
    # 3 korak dodeli instance najblizjemu centroidu
    # 4 korak izracunas kakovost
    # minimization manjsi fitnes je bolsi
    def _evaluate(self, x):
        print("Initial x.",x)
        clusters = []
        clusters_sum_dist = []
        all_clusters_dists = []
        # Step 1: Extract cluster centers from the solution vector 'x'
        cluster_centers = x.reshape((self.num_clusters, self.dimension))

        # Step 2: Compute distances from instances to cluster centers
        for clust_idx in range(self.num_clusters):
            cluster_center_dists = self.euclidean_distance(self.instances, cluster_centers[clust_idx])
            all_clusters_dists.append(np.array(cluster_center_dists))

        all_clusters_dists = np.array(all_clusters_dists)

        # Step 3: Assign instances to the nearest cluster center
        cluster_indices = np.argmin(all_clusters_dists, axis=0)

        for clust_idx in range(self.num_clusters):
            cluster_instances = np.where(cluster_indices == clust_idx)[0]
            clusters.append(cluster_instances)

            if len(cluster_instances) == 0:
                clusters_sum_dist.append(0)
            else:
                clusters_sum_dist.append(np.sum(all_clusters_dists[clust_idx, cluster_instances]))

        clusters_sum_dist = np.array(clusters_sum_dist)

        fitness = 1.0 / np.sum(clusters_sum_dist + 0.00000001)

        return fitness

