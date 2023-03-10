import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class PCADimensionReduction:
    def __init__(self, train_data, scaled_train_data, train_true_results, seed):
        self.train_data = train_data
        self.scaled_train_data = scaled_train_data
        self.train_true_results = train_true_results
        self.seed = seed

    def generate_scree_plots(self, components_amounts_list):
        self.__generate_scree_plots_for_data(
            self.train_data, components_amounts_list, "unscaled data"
        )

        self.__generate_scree_plots_for_data(
            self.scaled_train_data, components_amounts_list, "scaled data"
        )

    def __generate_scree_plots_for_data(
        self, data, components_amounts_list, data_title
    ):
        figure, axis = plt.subplots(1, len(components_amounts_list))
        figure.set_figheight(1.5)
        figure.set_figwidth(30)
        plt.subplots_adjust(top=3)
        for index, components_amount in enumerate(components_amounts_list):
            pca = PCA(n_components=components_amount, random_state=self.seed)
            pca.fit(data)
            per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

            axis[index].bar(x=range(1, len(per_var) + 1), height=per_var)
            axis[index].set_ylabel("Percentage of Explained Variance")
            axis[index].set_xlabel("Principal Component")
            axis[index].set_title(
                "Scree Plot of "
                + str(components_amount)
                + " components for "
                + data_title
            )
        plt.show()

    def display_pc1_pc2_plot(self):
        (
            unscaled_pca_cancer_cells_df,
            unscaled_pca_healthy_cell_df,
            unscaled_per_var,
        ) = self.__get_2d_pca_cancer_healthy_cells_indices_for_data(self.train_data)

        (
            scaled_pca_cancer_cells_df,
            scaled_pca_healthy_cell_df,
            scaled_per_var,
        ) = self.__get_2d_pca_cancer_healthy_cells_indices_for_data(
            self.scaled_train_data
        )

        figure, axis = plt.subplots(1, 2)
        figure.set_figheight(4)
        figure.set_figwidth(12)

        axis[0].scatter(
            unscaled_pca_healthy_cell_df.PC1,
            unscaled_pca_healthy_cell_df.PC2,
            c="lightblue",
            label="other cells",
        )
        axis[0].scatter(
            unscaled_pca_cancer_cells_df.PC1,
            unscaled_pca_cancer_cells_df.PC2,
            c="red",
            label="CTC",
        )
        axis[0].set_xlabel("PC1 - {0}%".format(unscaled_per_var[0]))
        axis[0].set_ylabel("PC2 - {0}%".format(unscaled_per_var[1]))
        axis[0].set_title("PCA projection for unscaled data")
        axis[0].legend()

        axis[1].scatter(
            scaled_pca_healthy_cell_df.PC1,
            unscaled_pca_healthy_cell_df.PC2,
            c="lightblue",
            label="other cells",
        )
        axis[1].scatter(
            scaled_pca_cancer_cells_df.PC1,
            scaled_pca_cancer_cells_df.PC2,
            c="red",
            label="CTC",
        )
        axis[1].set_xlabel("PC1 - {0}%".format(scaled_per_var[0]))
        axis[1].set_ylabel("PC2 - {0}%".format(scaled_per_var[1]))
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

        pca_healthy_cell_df = pd.DataFrame(
            pca_healthy_cells_data, columns=("PC1", "PC2")
        )
        pca_cancer_cells_df = pd.DataFrame(
            pca_cancer_cells_data, columns=("PC1", "PC2")
        )

        return pca_cancer_cells_df, pca_healthy_cell_df, per_var

    def get_most_important_variables_from_pc1(self, variables_amount):
        pca = PCA(random_state=self.seed)
        pca.fit(self.train_data)
        loading_scores = pd.Series(pca.components_[0], index=self.train_data.columns)
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        top_variables = sorted_loading_scores[0:variables_amount]
        return top_variables


def display_2d_plot(reducer, label, train_data, train_true_results, seed):
    data = reducer.fit_transform(train_data)

    cancer_cells_indices = np.where(train_true_results == 1)[0]
    healthy_cells_indices = np.where(train_true_results == 0)[0]

    healthy_cells_data = data[healthy_cells_indices]
    cancer_cells_data = data[cancer_cells_indices]

    healthy_cell_df = pd.DataFrame(healthy_cells_data, columns=("A", "B"))
    cancer_cells_df = pd.DataFrame(cancer_cells_data, columns=("A", "B"))

    plt.scatter(
        healthy_cell_df.A, healthy_cell_df.B, c="lightblue", label="other cells"
    )
    plt.scatter(cancer_cells_df.A, cancer_cells_df.B, c="red", label="CTC")
    plt.xlabel(label + "1")
    plt.ylabel(label + "2")
    plt.title(label + " projection of the CTC dataset", fontsize=24)

    plt.legend()
    plt.show()


def display_2d_plot_with_names(reducer, label, train_data, train_classes_names):
    data = reducer.fit_transform(train_data)

    cancer_cells_indices = np.where(train_classes_names == "CTC")[0]
    healthy_cells_indices = np.where(train_classes_names == "WBC")[0]
    mixed_cells_indices = np.where(train_classes_names == "CTC-WBC")[0]

    healthy_cells_data = data[healthy_cells_indices]
    cancer_cells_data = data[cancer_cells_indices]
    mixed_cells_data = data[mixed_cells_indices]

    healthy_cell_df = pd.DataFrame(healthy_cells_data, columns=("A", "B"))
    cancer_cells_df = pd.DataFrame(cancer_cells_data, columns=("A", "B"))
    mixed_cells_df = pd.DataFrame(mixed_cells_data, columns=("A", "B"))

    plt.scatter(
        healthy_cell_df.A, healthy_cell_df.B, c="lightblue", label="other cells"
    )
    plt.scatter(cancer_cells_df.A, cancer_cells_df.B, c="red", label="CTC")
    plt.scatter(
        mixed_cells_df.A, mixed_cells_df.B, c="green", label="mixed CTC and others"
    )
    plt.xlabel(label + "1")
    plt.ylabel(label + "2")
    plt.title(label + " projection of the CTC dataset", fontsize=24)

    plt.legend()
    plt.show()


def display_umap_2d_plot(
    train_data, train_true_results, seed, train_classes_names=None
):
    reducer = umap.UMAP(
        n_neighbors=3, random_state=seed
    )  # n_neighbors=3 yields the best results
    if train_classes_names is None:
        display_2d_plot(reducer, "UMAP", train_data, train_true_results)
    else:
        display_2d_plot_with_names(reducer, "UMAP", train_data, train_classes_names)


def display_tsne_2d_plot(
    train_data, train_true_results, seed, train_classes_names=None
):
    reducer = TSNE(learning_rate="auto", init="random", random_state=seed)
    if train_classes_names is None:
        display_2d_plot(reducer, "TSNE", train_data, train_true_results)
    else:
        display_2d_plot_with_names(reducer, "TSNE", train_data, train_classes_names)


def display_pca_2d_plot(train_data, train_true_results, seed, train_classes_names=None):
    reducer = PCA(n_components=2, random_state=seed)
    if train_classes_names is None:
        display_2d_plot(reducer, "PCA", train_data, train_true_results)
    else:
        display_2d_plot_with_names(reducer, "PCA", train_data, train_classes_names)
