import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random


class KMeansClustering:
    def __init__(self, true_results, seed):
        self.cancer_cells_indices = np.where(true_results == 1)[0]
        self.healthy_cells_indices = np.where(true_results == 0)[0]
        self.seed = seed

    def k_means_clustering_for_different_k(self, data, k_array, pca=0):
        clustering_percentages_list = []
        clusters_numbers_list = []
        pca_data = None

        if pca:
            pca = PCA(n_components=pca, random_state=self.seed)
            pca.fit(data)
            pca_data = pca.transform(data)

        for k in k_array:
            k_means_object = KMeans(n_clusters=k, random_state=self.seed)

            if pca:
                k_means_object.fit(pca_data)
            else:
                k_means_object.fit(data)

            clustering_results = k_means_object.labels_
            cancer_cells_clustering = clustering_results[self.cancer_cells_indices]
            clusters_number, clustering_percentage = KMeansClustering.ctc_clustering_metrics(cancer_cells_clustering)
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

    @staticmethod
    def ctc_clustering_metrics(array):
        counter = collections.Counter(array)
        most_frequent_number_tuple = counter.most_common(1)
        max_occurrence = most_frequent_number_tuple[0][1]

        return len(counter), (max_occurrence / len(array)) * 100

    def display_2d_pca_clustering(self, data, k_array):
        pca = PCA(n_components=2, random_state=self.seed)
        pca.fit(data)
        pca_data = pca.transform(data)

        figure, axis = plt.subplots(1, len(k_array))
        figure.set_figheight(5)
        figure.set_figwidth(20)

        random.seed(self.seed)
        colors = []
        for i in range(max(k_array)):
            colors.append("#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))

        for index, k in enumerate(k_array):
            k_means_object = KMeans(n_clusters=k, random_state=self.seed)
            k_means_object.fit(pca_data)

            labels = k_means_object.labels_
            ctc_labels = labels[self.cancer_cells_indices]
            ctc_data = pca_data[self.cancer_cells_indices]
            healthy_labels = labels[self.healthy_cells_indices]
            healthy_data = pca_data[self.healthy_cells_indices]

            for i in range(k):
                axis[index].scatter(healthy_data[healthy_labels == i, 0], healthy_data[healthy_labels == i, 1], label=i,
                                    c=colors[i])
                axis[index].scatter(ctc_data[ctc_labels == i, 0], ctc_data[ctc_labels == i, 1], label=i,
                                    edgecolors='red', linewidths=0.7, c=colors[i])

            axis[index].set_xlabel('PC1')
            axis[index].set_ylabel('PC2')
            axis[index].set_title('Clustering for 2D PCA with ' + str(k) + ' clusters')

        plt.show()
