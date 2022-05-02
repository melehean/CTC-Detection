import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def k_means_clustering_for_different_k(data, true_results, k_array, seed, pca=0):
    cancer_cells_indices = np.where(true_results == 1)[0]

    clustering_percentages_list = []
    clusters_numbers_list = []
    pca_data = None

    if pca:
        pca = PCA(n_components=pca, random_state=seed)
        pca.fit(data)
        pca_data = pca.transform(data)

    for k in k_array:
        k_means_object = KMeans(n_clusters=k, random_state=seed)

        if pca:
            k_means_object.fit(pca_data)
        else:
            k_means_object.fit(data)

        clustering_results = k_means_object.labels_
        cancer_cells_clustering = clustering_results[cancer_cells_indices]
        clusters_number, clustering_percentage = ctc_clustering_metrics(cancer_cells_clustering)
        clustering_percentages_list.append(clustering_percentage)
        clusters_numbers_list.append(clusters_number)

    figure, axis = plt.subplots(1, 2)
    figure.set_figheight(5)
    figure.set_figwidth(15)

    axis[0].plot(k_array, clustering_percentages_list)
    axis[0].set_ylabel('Maximum percentage of 1 cluster occurrence for CTCs')
    axis[0].set_xlabel('Number of clusters')
    axis[0].set_title('Maximum percentage of 1 cluster occurrence for CTCs for given algorithm clusters')

    axis[1].plot(k_array, clusters_numbers_list)
    axis[1].set_ylabel('Number of different clusters for CTCs')
    axis[1].set_xlabel('Number of clusters')
    axis[1].set_title('Number of different clusters for CTCs for given algorithm clusters')

    plt.show()


def ctc_clustering_metrics(array):
    counter = collections.Counter(array)
    most_frequent_number_tuple = counter.most_common(1)
    max_occurrence = most_frequent_number_tuple[0][1]

    return len(counter), (max_occurrence / len(array)) * 100
