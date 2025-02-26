import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_clusters(data, labels, centroids, num_clusters, title):
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('plasma', num_clusters)

    pca_model = PCA(n_components=2).fit(data)
    data_2d = pca_model.transform(data)
    centroids_2d = pca_model.transform(centroids)

    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap=cmap)

    for cluster_num in range(num_clusters):
        cluster_centroid = centroids_2d[cluster_num]
        plt.scatter(cluster_centroid[0], cluster_centroid[1], s=100, marker='o',
                    facecolor='red', edgecolor='black',
                    label=f'Centroid: {cluster_num}')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
