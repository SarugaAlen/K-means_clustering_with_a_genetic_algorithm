import matplotlib.pyplot as plt


def visualize_clusters(data, labels, centroids, num_clusters, title):
    plt.figure(figsize=(8, 6))
    # Define color map based on the number of clusters
    cmap = plt.cm.get_cmap('viridis', num_clusters)

    # Plot data points for each pair of features
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            plt.scatter(data[:, i], data[:, j], c=labels, cmap=cmap)

    # Plot cluster centers with circles colored according to the cluster they belong to
    for cluster_num in range(num_clusters):
        cluster_color = cmap(cluster_num)
        plt.scatter(centroids[cluster_num, 0], centroids[cluster_num, 1],
                    s=100, marker='o', c=[cluster_color], edgecolor='black', label=f'Centroid: {cluster_num}')

    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Feature')
    plt.legend()
    plt.show()

