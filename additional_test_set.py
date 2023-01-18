import pandas as pd
import numpy as np

from dimension_reduction import PCADimensionReduction


class AdditionalTestSet:
    def __init__(
        self,
        train_data_path,
        test_data_path,
        train_sample_info_path,
        test_sample_info_path,
    ):
        self.train_data = None
        self.test_data = None
        self.test_classes = None
        self.train_classes = None

        self.load_train_data(train_data_path)
        self.load_test_data(test_data_path)
        self.prepare_train_classes(train_sample_info_path)
        self.prepare_test_classes(test_sample_info_path)

    def load_train_data(self, file_path):
        self.train_data = pd.read_csv(file_path, sep="\t")
        self.train_data = self.train_data.transpose()
        self.train_data = self.train_data.loc[:, (self.train_data != 0).any(axis=0)]

    def load_test_data(self, file_path):
        self.test_data = pd.read_csv(file_path, sep="\t")
        self.test_data = self.test_data.transpose()
        self.test_data = self.test_data[self.train_data.columns]

    def prepare_test_classes_names(self, sample_info):
        self.test_classes_names = sample_info.set_index("Sample")
        self.test_classes_names = self.test_classes_names.drop(columns=["Group"])
        self.test_classes_names = self.test_classes_names.squeeze()

    def prepare_test_classes_numbers(self, sample_info):
        self.test_classes = sample_info.set_index("Sample")
        self.test_classes = self.test_classes.drop(columns=["Group"])
        self.test_classes[self.test_classes["GroupDetailed"] == "WBC"] = 0
        self.test_classes[
            (self.test_classes["GroupDetailed"] == "CTC")
            | (self.test_classes["GroupDetailed"] == "CTC-WBC")
        ] = 1
        self.test_classes = self.test_classes.astype(int)
        self.test_classes = self.test_classes.squeeze()

    def prepare_train_classes(self, file_path):
        self.train_classes = pd.read_csv(file_path, sep="\t")
        self.train_classes = self.train_classes.set_index("Sample")
        self.train_classes[self.train_classes["Group"] == "WBC"] = 0
        self.train_classes[self.train_classes["Group"] == "CTC"] = 1
        self.train_classes = self.train_classes.astype(int)
        self.train_classes = self.train_classes.squeeze()

    def prepare_test_classes(self, file_path):
        sample_info = pd.read_csv(
            "data/CTC_new_28_12_2022/sampleInfoTest.tsv", sep="\t"
        )
        self.prepare_test_classes_numbers(sample_info)
        self.prepare_test_classes_names(sample_info)

    def summary(
        self,
    ):
        healthy_cells_train_indices = np.where(self.train_classes == 0)[0]
        cancer_cells_train_indices = np.where(self.train_classes == 1)[0]

        healthy_cells_test_indices = np.where(self.test_classes_names == "WBC")[0]
        cancer_cells_test_indices = np.where(self.test_classes_names == "CTC")[0]
        mix_cells_test_indices = np.where(self.test_classes_names == "CTC-WBC")[0]

        print(f"Total cells number in train data: {len(self.train_classes)}")
        print(f"CTC cells number in train data: {len(cancer_cells_train_indices)}")
        print(f"WBC cells number in train data: {len(healthy_cells_train_indices)}\n")

        print(f"Total cells number in test data: {len(self.test_classes)}")
        print(f"CTC cells number in test data: {len(cancer_cells_test_indices)}")
        print(f"WBC cells number in test data: {len(healthy_cells_test_indices)}")
        print(f"CTC-WBC cells number in test data: {len(mix_cells_test_indices)}")

    @staticmethod
    def cut_data_by_mean(train_data, test_data, threshold):
        columns_bool_values = train_data.mean() > threshold  # 0.6
        cut_by_mean_train_data = train_data[train_data.columns[columns_bool_values]]
        cut_by_mean_test_data = test_data[cut_by_mean_train_data.columns]
        return cut_by_mean_train_data, cut_by_mean_test_data

    def cut_data_by_mean_with_classes(self, train_data, test_data, threshold):
        healthy_cells_train_indices = np.where(self.train_classes == 0)[0]
        healthy_train_data = train_data.iloc[healthy_cells_train_indices]

        cancer_cells_train_indices = np.where(self.train_classes == 1)[0]
        cancer_train_data = train_data.iloc[cancer_cells_train_indices]

        columns_bool_values = (healthy_train_data.mean() > threshold) & (
            cancer_train_data.mean() > threshold
        )  # 0.6
        cut_by_mean_train_data = train_data[train_data.columns[columns_bool_values]]
        cut_by_mean_test_data = test_data[cut_by_mean_train_data.columns]
        return cut_by_mean_train_data, cut_by_mean_test_data

    @staticmethod
    def cut_data_by_max(train_data, test_data, threshold):
        columns_bool_values = train_data.max() > threshold  # 7
        cut_by_max_train_data = train_data[train_data.columns[columns_bool_values]]
        cut_by_max_test_data = test_data[cut_by_max_train_data.columns]
        return cut_by_max_train_data, cut_by_max_test_data

    def reduced_data_by_pca(self, train_data, test_data, feature_number):
        pca_object = PCADimensionReduction(
            train_data, train_data, self.train_classes, 42
        )
        pca_variables = pca_object.get_most_important_variables_from_pc1(
            feature_number
        )  # 60
        pca_reduced_train_data = train_data[pca_variables.index]
        pca_reduced_test_data = test_data[pca_variables.index]

    def reduced_data_by_mean_fold_change(self, train_data, test_data, threshold):
        healthy_cells_train_indices = np.where(self.train_classes == 0)[0]
        healthy_train_data = train_data.iloc[healthy_cells_train_indices]

        cancer_cells_train_indices = np.where(self.train_classes == 1)[0]
        cancer_train_data = train_data.iloc[cancer_cells_train_indices]

        mean_fold_change = np.where(
            cancer_train_data.mean() > healthy_train_data.mean(),
            cancer_train_data.mean() / healthy_train_data.mean(),
            healthy_train_data.mean() / cancer_train_data.mean(),
        )

        columns_bool_values = mean_fold_change
        cut_by_mean_train_data = train_data[train_data.columns[columns_bool_values]]
        cut_by_mean_test_data = test_data[cut_by_mean_train_data.columns]
        return cut_by_mean_train_data, cut_by_mean_test_data
