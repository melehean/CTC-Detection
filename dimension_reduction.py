from sklearn.decomposition import PCA
import umap

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCADimensionReduction:
    def __init__(self, train_data, scaled_train_data, train_true_results, seed):
        self.train_data = train_data
        self.scaled_train_data = scaled_train_data
        self.train_true_results = train_true_results
        self.seed = seed

    def generate_scree_plots(self, components_amounts_list):
        self.__generate_scree_plots_for_data(self.train_data, components_amounts_list, "unscaled data")

        self.__generate_scree_plots_for_data(self.scaled_train_data, components_amounts_list, "scaled data")

    def __generate_scree_plots_for_data(self, data, components_amounts_list, data_title):
        figure, axis = plt.subplots(1, len(components_amounts_list))
        figure.set_figheight(1.5)
        figure.set_figwidth(30)
        plt.subplots_adjust(top=3)
        for index, components_amount in enumerate(components_amounts_list):
            pca = PCA(n_components=components_amount, random_state=self.seed)
            pca.fit(data)
            per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

            axis[index].bar(x=range(1, len(per_var) + 1), height=per_var)
            axis[index].set_ylabel('Percentage of Explained Variance')
            axis[index].set_xlabel('Principal Component')
            axis[index].set_title('Scree Plot of ' + str(components_amount) + ' components for ' + data_title)
        plt.show()

    def display_pc1_pc2_plot(self):
        unscaled_pca_cancer_cells_df, unscaled_pca_healthy_cell_df, unscaled_per_var = \
            self.__get_2d_pca_cancer_healthy_cells_indices_for_data(self.train_data)

        scaled_pca_cancer_cells_df, scaled_pca_healthy_cell_df, scaled_per_var = \
            self.__get_2d_pca_cancer_healthy_cells_indices_for_data(self.scaled_train_data)

        figure, axis = plt.subplots(1, 2)
        figure.set_figheight(4)
        figure.set_figwidth(12)

        axis[0].scatter(unscaled_pca_healthy_cell_df.PC1, unscaled_pca_healthy_cell_df.PC2,
                        c='lightblue', label='other cells')
        axis[0].scatter(unscaled_pca_cancer_cells_df.PC1, unscaled_pca_cancer_cells_df.PC2,
                        c='red', label='CTC')
        axis[0].set_xlabel('PC1 - {0}%'.format(unscaled_per_var[0]))
        axis[0].set_ylabel('PC2 - {0}%'.format(unscaled_per_var[1]))
        axis[0].set_title("PCA projection for unscaled data")
        axis[0].legend()

        axis[1].scatter(scaled_pca_healthy_cell_df.PC1, unscaled_pca_healthy_cell_df.PC2,
                        c='lightblue', label='other cells')
        axis[1].scatter(scaled_pca_cancer_cells_df.PC1, scaled_pca_cancer_cells_df.PC2,
                        c='red', label='CTC')
        axis[1].set_xlabel('PC1 - {0}%'.format(scaled_per_var[0]))
        axis[1].set_ylabel('PC2 - {0}%'.format(scaled_per_var[1]))
        axis[1].set_title("PCA projection for scaled data")
        axis[1].legend()

        plt.show()

    def __get_2d_pca_cancer_healthy_cells_indices_for_data(self, data):
        pca = PCA(n_components=2, random_state=self.seed)
        pca.fit(data)
        pca_data = pca.transform(data)
        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

        cancer_cells_indices = np.where(self.train_true_results == 1)[0]
        healthy_cells_indices = np.where(self.train_true_results == 0)[0]

        pca_healthy_cells_data = pca_data[healthy_cells_indices]
        pca_cancer_cells_data = pca_data[cancer_cells_indices]

        pca_healthy_cell_df = pd.DataFrame(pca_healthy_cells_data, columns=('PC1', 'PC2'))
        pca_cancer_cells_df = pd.DataFrame(pca_cancer_cells_data, columns=('PC1', 'PC2'))

        return pca_cancer_cells_df, pca_healthy_cell_df, per_var

    def get_most_important_variables_from_pc1(self, variables_amount):
        pca = PCA(random_state=self.seed)
        pca.fit(self.train_data)
        loading_scores = pd.Series(pca.components_[0], index=self.train_data.columns)
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        top_variables = sorted_loading_scores[0:variables_amount]
        return top_variables


def display_umap_2d_plot(train_data, train_true_results, seed):
    reducer = umap.UMAP(n_neighbors=3, random_state=seed)  # n_neighbors=3 yields the best results
    umap_data = reducer.fit_transform(train_data)

    cancer_cells_indices = np.where(train_true_results == 1)[0]
    healthy_cells_indices = np.where(train_true_results == 0)[0]

    umap_healthy_cells_data = umap_data[healthy_cells_indices]
    umap_cancer_cells_data = umap_data[cancer_cells_indices]

    umap_healthy_cell_df = pd.DataFrame(umap_healthy_cells_data, columns=('UMAP1', 'UMAP2'))
    umap_cancer_cells_df = pd.DataFrame(umap_cancer_cells_data, columns=('UMAP1', 'UMAP2'))

    plt.scatter(umap_healthy_cell_df.UMAP1, umap_healthy_cell_df.UMAP2, c='lightblue', label='other cells')
    plt.scatter(umap_cancer_cells_df.UMAP1, umap_cancer_cells_df.UMAP2, c='red', label='CTC')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP projection of the CTC dataset', fontsize=24)

    plt.legend()
    plt.show()
