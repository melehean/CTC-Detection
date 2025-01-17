import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Data:
    def __init__(self, data_filepath, true_results_filepath):
        self.test_true_results = None
        self.train_true_results = None
        self.test_data = None
        self.train_data = None
        self.train_indices = None
        self.test_indices = None

        self.data = pd.read_csv(data_filepath, sep="\t")
        self.data = self.data.transpose()
        self.true_results = pd.read_csv(true_results_filepath, sep="\t")
        self.data_amount = len(self.data.index)

        self.cancer_train_data = None
        self.cancer_test_data = None
        self.healthy_train_data = None
        self.healthy_test_data = None

    def generate_train_test_split(self):
        indices = np.arange(self.data_amount)
        (
            self.train_data,
            self.test_data,
            self.train_true_results,
            self.test_true_results,
            self.train_indices,
            self.test_indices,
        ) = train_test_split(
            self.data,
            self.true_results,
            indices,
            test_size=0.3,
            random_state=42,
            shuffle=True,
            stratify=self.true_results,
        )

        np.save("train_indices.npy", self.train_indices)
        np.save("test_indices.npy", self.test_indices)

    def load_train_test_split(self, train_indices_filepath, test_indices_filepath):
        self.train_indices = np.load(train_indices_filepath)
        self.test_indices = np.load(test_indices_filepath)

        self.train_data = self.data.iloc[self.train_indices]
        self.test_data = self.data.iloc[self.test_indices]

        self.train_true_results = self.true_results.iloc[self.train_indices]
        self.test_true_results = self.true_results.iloc[self.test_indices]

        return (
            self.train_data,
            self.test_data,
            self.train_true_results,
            self.test_true_results,
        )

    def get_converted_train_test_split(self):
        return (
            np.array(self.train_data, np.float32),
            np.array(self.test_data, np.float32),
            np.array(self.train_true_results, np.int32),
            np.array(self.test_true_results, np.int32),
        )

    def load_train_test_healthy_cancer_data(self):
        cancer_train_indices = np.where(self.train_true_results == 1)[0]
        healthy_train_indices = np.where(self.train_true_results == 0)[0]

        cancer_test_indices = np.where(self.test_true_results == 1)[0]
        healthy_test_indices = np.where(self.test_true_results == 0)[0]

        self.cancer_train_data = self.train_data.iloc[cancer_train_indices]
        self.cancer_test_data = self.test_data.iloc[cancer_test_indices]
        self.healthy_train_data = self.train_data.iloc[healthy_train_indices]
        self.healthy_test_data = self.test_data.iloc[healthy_test_indices]

        return (
            self.healthy_train_data,
            self.cancer_train_data,
            self.healthy_test_data,
            self.cancer_test_data,
        )

    def get_all_data(self):
        return self.data

    def get_all_true_results(self):
        return self.true_results

    def get_scaled_train_test_data(self):
        scaler = StandardScaler().fit(self.train_data)
        scaled_train_data = pd.DataFrame(
            scaler.transform(self.train_data), columns=self.train_data.columns
        )
        scaled_test_data = pd.DataFrame(
            scaler.transform(self.test_data), columns=self.test_data.columns
        )
        return scaled_train_data, scaled_test_data

    def get_cut_by_max_train_test_data(self, threshold):
        columns_bool_values = self.train_data.max() > threshold
        cut_by_max_train_data = self.train_data[
            self.train_data.columns[columns_bool_values]
        ]
        cut_by_max_test_data = self.test_data[
            self.test_data.columns[columns_bool_values]
        ]
        return cut_by_max_train_data, cut_by_max_test_data

    def summary(
        self,
    ):
        healthy_cells_train_indices = np.where(self.train_true_results == 0)[0]
        cancer_cells_train_indices = np.where(self.train_true_results == 1)[0]

        healthy_cells_test_indices = np.where(self.test_true_results == 0)[0]
        cancer_cells_test_indices = np.where(self.test_true_results == 1)[0]

        print(f"Total cells number in train data: {len(self.train_true_results)}")
        print(f"CTC cells number in train data: {len(cancer_cells_train_indices)}")
        print(f"WBC cells number in train data: {len(healthy_cells_train_indices)}\n")

        print(f"Total cells number in test data: {len(self.test_true_results)}")
        print(f"CTC cells number in test data: {len(cancer_cells_test_indices)}")
        print(f"WBC cells number in test data: {len(healthy_cells_test_indices)}")

    @staticmethod
    def cut_data_by_mean(train_data, test_data, threshold):
        columns_bool_values = train_data.mean() > threshold  # 0.6
        cut_by_mean_train_data = train_data[train_data.columns[columns_bool_values]]
        cut_by_mean_test_data = test_data[cut_by_mean_train_data.columns]
        return cut_by_mean_train_data, cut_by_mean_test_data

    @staticmethod
    def cut_data_by_max(train_data, test_data, threshold):
        columns_bool_values = train_data.max() > threshold  # 7
        cut_by_max_train_data = train_data[train_data.columns[columns_bool_values]]
        cut_by_max_test_data = test_data[cut_by_max_train_data.columns]
        return cut_by_max_train_data, cut_by_max_test_data

    def scaled_data_by_healthy_train(self, data):
        scaler = StandardScaler().fit(self.healthy_train_data)
        scaled_data = pd.DataFrame(
            scaler.transform(data),
            columns=data.columns,
        )
        return scaled_data

    def get_scaled_healthy_train_test_cancer_data(self):
        scaled_healthy_train_data = self.scaled_data_by_healthy_train(
            self.healthy_train_data
        )
        scaled_healthy_test_data = self.scaled_data_by_healthy_train(
            self.healthy_test_data
        )
        scaled_cancer_train_data = self.scaled_data_by_healthy_train(
            self.cancer_train_data
        )
        scaled_cancer_test_data = self.scaled_data_by_healthy_train(
            self.cancer_test_data
        )
        return (
            scaled_healthy_train_data,
            scaled_healthy_test_data,
            scaled_cancer_train_data,
            scaled_cancer_test_data,
        )

    def cut_by_max_healthy_data(self, data, threshold):
        columns_bool_values = self.healthy_train_data.max() > threshold
        cut_by_max_data = data[data.columns[columns_bool_values]]
        return cut_by_max_data

    def get_cut_by_max_healthy_train_test_cancer_data(self, threshold):
        cut_by_max_healthy_train_data = self.cut_by_max_healthy_data(
            self.healthy_train_data, threshold
        )
        cut_by_max_healthy_test_data = self.cut_by_max_healthy_data(
            self.healthy_test_data, threshold
        )
        cut_by_max_cancer_train_data = self.cut_by_max_healthy_data(
            self.cancer_train_data, threshold
        )
        cut_by_max_cancer_test_data = self.cut_by_max_healthy_data(
            self.cancer_test_data, threshold
        )
        return (
            cut_by_max_healthy_train_data,
            cut_by_max_healthy_test_data,
            cut_by_max_cancer_train_data,
            cut_by_max_cancer_test_data,
        )

    def cut_data_by_mean_with_classes(self, train_data, test_data, threshold):
        healthy_cells_train_indices = np.where(self.train_true_results == 0)[0]
        healthy_train_data = train_data.iloc[healthy_cells_train_indices]

        cancer_cells_train_indices = np.where(self.train_true_results == 1)[0]
        cancer_train_data = train_data.iloc[cancer_cells_train_indices]

        columns_bool_values = (healthy_train_data.mean() > threshold) & (
            cancer_train_data.mean() > threshold
        )  # 0.6
        cut_by_mean_train_data = train_data[train_data.columns[columns_bool_values]]
        cut_by_mean_test_data = test_data[cut_by_mean_train_data.columns]
        return cut_by_mean_train_data, cut_by_mean_test_data

    def get_filtered_by_zero_data(self, train_data, test_data, threshold):
        healthy_cells_train_indices = np.where(self.train_true_results == 0)[0]
        healthy_train_data = train_data.iloc[healthy_cells_train_indices]

        filtered_by_zero_values_train_data = train_data
        filtered_by_zero_values_test_data = test_data
        number_of_deleted_columns = 0

        for (columnName, columnData) in train_data.iteritems():
            num_of_zero_values = (
                train_data.iloc[healthy_cells_train_indices][columnName] == 0
            ).sum()
            if num_of_zero_values > threshold * len(train_data[columnName]):
                filtered_by_zero_values_train_data.drop(
                    columns=[columnName], axis=1, inplace=True
                )
                filtered_by_zero_values_test_data.drop(
                    columns=[columnName], axis=1, inplace=True
                )
                number_of_deleted_columns += 1

        print(f"Number of deleted columns: {number_of_deleted_columns}")

        healthy_cells_train_indices = np.where(self.train_true_results == 0)[0]
        filtered_by_zero_healthy_train_data = filtered_by_zero_values_train_data.iloc[
            healthy_cells_train_indices
        ]

        healthy_cells_test_indices = np.where(self.test_true_results == 0)[0]
        filtered_by_zero_healthy_test_data = filtered_by_zero_values_test_data.iloc[
            healthy_cells_test_indices
        ]

        cancer_cells_train_indices = np.where(self.train_true_results == 1)[0]
        cancer_train_data = filtered_by_zero_values_train_data.iloc[
            cancer_cells_train_indices
        ]

        cancer_cells_test_indices = np.where(self.test_true_results == 1)[0]
        cancer_test_data = filtered_by_zero_values_test_data.iloc[
            cancer_cells_test_indices
        ]

        return (
            filtered_by_zero_values_train_data,
            filtered_by_zero_values_test_data,
            filtered_by_zero_healthy_train_data,
            filtered_by_zero_healthy_test_data,
            cancer_train_data,
            cancer_test_data,
        )
