import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, data_filepath, true_results_filepath):
        self.test_true_results = None
        self.train_true_results = None
        self.test_data = None
        self.train_data = None
        self.train_indices = None
        self.test_indices = None
        self.data = pd.read_csv(data_filepath, sep='\t')
        self.data = self.data.transpose()
        self.true_results = pd.read_csv(true_results_filepath, sep='\t')
        self.data_amount = len(self.data.index)

    def generate_train_test_split(self):
        indices = np.arange(self.data_amount)
        self.train_data, self.test_data, self.train_true_results, self.test_true_results, self.train_indices, \
        self.test_indices = \
            train_test_split(self.data, self.true_results, indices, test_size=30, random_state=42, shuffle=True,
                             stratify=self.true_results)

        np.save("train_indices.npy", self.train_indices)
        np.save("test_indices.npy", self.test_indices)

    def load_train_test_split(self, train_indices_filepath, test_indices_filepath):
        self.train_indices = np.load(train_indices_filepath)
        self.test_indices = np.load(test_indices_filepath)

        self.train_data = self.data.iloc[self.train_indices]
        self.test_data = self.data.iloc[self.test_indices]

        self.train_true_results = self.true_results.iloc[self.train_indices]
        self.test_true_results = self.true_results.iloc[self.test_indices]

        return self.train_data, self.test_data, self.train_true_results, self.test_true_results

    def get_all_data(self):
        return self.data

    def get_all_true_results(self):
        return self.true_results
