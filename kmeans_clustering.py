import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def perform_kmeans_clustering(data, n_clusters=3):
    os.environ['OMP_NUM_THREADS'] = '1'

    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init='auto', max_iter=300)
    kmeans.fit(data)
    labels = kmeans.labels_
    cmap = plt.cm.get_cmap('viridis', n_clusters)

    plt.figure(figsize=(8, 6))
    for i in range(data.shape[1]):
        for j in range(i+1, data.shape[1]):
            plt.scatter(data[:, i], data[:, j], c=labels, cmap=cmap, label=f'Feature {i+1} vs Feature {j+1}')

    for cluster_num in range(n_clusters):
        cluster_color = cmap(cluster_num)
        plt.scatter(kmeans.cluster_centers_[cluster_num, 0], kmeans.cluster_centers_[cluster_num, 1],
                    s=100, marker='o', c=[cluster_color], edgecolor='black', label=f'Središče {cluster_num + 1}')

    plt.title('K-Means Gručenje')
    plt.xlabel('Feature')
    plt.ylabel('Feature')
    plt.legend()
    plt.show()
