import matplotlib.pyplot as plt


def visualize_clusters(data, labels, centroids, num_clusters, title):
    plt.figure(figsize=(8, 6))
    # Define color map based on the number of clusters
    cmap = plt.cm.get_cmap('viridis', num_clusters)

    # Plot data points
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap)

    # Plot cluster centers with the same color as the cluster they belong to
    for cluster_num in range(num_clusters):
        cluster_color = cmap(cluster_num)
        cluster_centroid = centroids[cluster_num]
        plt.scatter(cluster_centroid[0], cluster_centroid[1], s=100, marker='o',
                    facecolor=cluster_color, edgecolor='black',
                    label=f'Centroid: {cluster_num}')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
