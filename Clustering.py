import numpy
import numpy as np
from niapy.problems import Problem
from sklearn.metrics import pairwise_distances_argmin_min

class Clustering(Problem):
    def __init__(self, num_clusters, dimension, lower=0, upper=1, instances=None, *args, **kwargs):
        super().__init__(dimension=dimension, lower=lower, upper=upper, *args, **kwargs)
        self.num_clusters = num_clusters
        self.instances = instances

        def _evaluate(self, x): # x = [5.5, 8, 8, 3.5] -> [[5.5, 8],[8 , 3.5]]
            x = np.array(x)
            cluster_centers = x.reshape(self.num_clusters, self.dimension)

            #fitness = 1.0 / np.sum(np.min(distances) + 0.00000001)
            #return fitness
            return 0.0

    # 1 korak iz x dobis lokacijen za vse centroide

    # 2 korak za vsako instanco izracunas razdaljo do vseh centroidov
    # 3 korak dodeli instance najblizjemu centroidu
    # 4 korak izracunas kakovost
    # minimization manjsi fitnes je bolsi

    def _get_centroids(self, x):
        # Calculate the number of columns for the centroids array
        num_cols = len(x) // self.num_clusters

        # Reshape the input array into a 2D array with num_clusters rows and num_cols columns
        centroids = x.reshape((self.num_clusters, num_cols))

        return centroids

    def euclidean_distance(self, X, Y):
        # Calculate the squared differences between each element of X and Y
        squared_diff = np.power(X - Y, 2)

        # Sum the squared differences along axis=1 (across columns) to get the sum of squared differences for each row
        sum_squared_diff = np.sum(squared_diff, axis=1)

        # Take the square root of the summed squared differences to get the Euclidean distance
        distances = np.sqrt(sum_squared_diff)

        return distances
