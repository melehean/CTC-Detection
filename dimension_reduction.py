from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class PCADimensionReduction:
    def __init__(self, train_data, train_true_results):
        self.train_data = train_data
        self.train_true_results = train_true_results

    def generate_scree_plots(self, components_amounts_list):
        figure, axis = plt.subplots(1, len(components_amounts_list))
        figure.set_figheight(1)
        figure.set_figwidth(20)
        plt.subplots_adjust(top=3)
        for index, components_amount in enumerate(components_amounts_list):
            pca = PCA(n_components=components_amount)
            pca.fit(self.train_data)
            per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)

            axis[index].bar(x=range(1, len(per_var)+1), height=per_var)
            axis[index].set_ylabel('Percentage of Explained Variance')
            axis[index].set_xlabel('Principal Component')
            axis[index].set_title('Scree Plot of ' + str(components_amount) + ' components')
        plt.show()

    def display_pc1_pc2_plot(self):
        pca = PCA(n_components=2)
        pca.fit(self.train_data)
        pca_data = pca.transform(self.train_data)
        per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)

        cancer_cells_indices = np.where(self.train_true_results == 1)[0]
        healthy_cells_indices = np.where(self.train_true_results == 0)[0]

        pca_healthy_cells_data = pca_data[healthy_cells_indices]
        pca_cancer_cells_data = pca_data[cancer_cells_indices]

        pca_healthy_cell_df = pd.DataFrame(pca_healthy_cells_data, columns=('PC1', 'PC2'))
        pca_cancer_cells_df = pd.DataFrame(pca_cancer_cells_data, columns=('PC1', 'PC2'))

        plt.scatter(pca_healthy_cell_df.PC1, pca_healthy_cell_df.PC2, c='lightblue', label='other cells')
        plt.scatter(pca_cancer_cells_df.PC1, pca_cancer_cells_df.PC2, c='red', label='CTC')
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))

        plt.legend()
        plt.show()

    def get_most_important_variables_from_pc1(self, variables_amount):
        pca = PCA()
        pca.fit(self.train_data)
        loading_scores = pd.Series(pca.components_[0], index=self.train_data.columns)
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        top_variables = sorted_loading_scores[0:variables_amount]
        return top_variables
