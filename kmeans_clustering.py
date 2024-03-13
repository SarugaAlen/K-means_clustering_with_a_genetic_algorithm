import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def perform_kmeans_clustering(data, n_clusters=3):
    os.environ['OMP_NUM_THREADS'] = '1'

    ''' Inicializacija KMeans gručenja '''
    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init='auto', max_iter=300, random_state=42)
    kmeans.fit(data)

    ''' Pridobivanje razredov '''
    labels = kmeans.labels_

    '''
    V tem razdelku se za vse pare funkcij ustvari razpršeni diagram, pri čemer se podatkovne 
    točke obarvajo glede na dodeljene oznake gruč. 
    Prav tako prikaže središča gruč z dosledno barvo in črnim robom.
    '''

    cmap = plt.cm.get_cmap('viridis', n_clusters)

    ''' Izris gruč '''
    plt.figure(figsize=(8, 6))

    for i in range(data.shape[1]):
        for j in range(i+1, data.shape[1]):
            plt.scatter(data[:, i], data[:, j], c=labels, cmap=cmap, label=f'Feature {i+1} vs Feature {j+1}')

    ''' Izris središč gruč '''
    for cluster_num in range(n_clusters):
        cluster_color = cmap(cluster_num)
        plt.scatter(kmeans.cluster_centers_[cluster_num, 0], kmeans.cluster_centers_[cluster_num, 1],
                    s=100, marker='o', c=[cluster_color], edgecolor='black', label=f'Središče {cluster_num + 1}')

    plt.title('K-Means Gručenje')
    plt.xlabel('Feature')
    plt.ylabel('Feature')
    plt.legend()
    plt.show()
