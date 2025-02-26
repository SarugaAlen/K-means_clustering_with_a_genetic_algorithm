import numpy as np
from niapy.problems import Problem


def euclidean_distance(instance, cluster_center):
    return np.sqrt(np.sum(np.power(instance - cluster_center, 2), axis=1))


class Clustering(Problem):
    def __init__(self, num_clusters, num_features, dimension, instances, lower=0, upper=1, *args, **kwargs):
        super().__init__(dimension=dimension, lower=lower, upper=upper, *args, **kwargs)
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.instances = instances
        self.cluster_indices = None

    def _evaluate(self, x):
        all_clusters_dists = []
        clusters = []
        clusters_sum_dist = []

        x = x.reshape(self.num_clusters, self.num_features)

        for clust_idx in range(self.num_clusters):
            cluster_center_dist = euclidean_distance(self.instances, x[clust_idx])
            all_clusters_dists.append(np.array(cluster_center_dist))

        all_clusters_dists = np.array(all_clusters_dists)

        cluster_indices = np.argmin(all_clusters_dists, axis=0)
        self.cluster_indices = np.array(cluster_indices)

        for clust_idx in range(self.num_clusters):
            clusters.append(np.where(cluster_indices == clust_idx)[0])
            if len(clusters[clust_idx]) == 0:
                clusters_sum_dist.append(0)
            else:
                clusters_sum_dist.append(np.sum(all_clusters_dists[clust_idx, clusters[clust_idx]]))

        clusters_sum_dist = np.array(clusters_sum_dist)

        fitness = np.sum(clusters_sum_dist)

        return fitness
