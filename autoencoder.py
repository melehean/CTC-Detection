import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)


class Autoencoder:
    def __init__(
        self,
        model,
        healthy_train_data,
        healthy_test_data,
        cancer_train_data,
        cancer_test_data,
    ):
        self.model = model
        self.healthy_train_data = healthy_train_data
        self.healthy_test_data = healthy_test_data
        self.cancer_train_data = cancer_train_data
        self.cancer_test_data = cancer_test_data
        self.cancer_train_rmse = None
        self.cancer_test_rmse = None
        self.healthy_test_rmse = None
        self.healthy_train_rmse = None

        self.threshold = self._calculate_threshold()

    def _calculate_rmse_list(self):
        self.cancer_train_rmse = self._calculate_rmse_for_data(self.cancer_train_data)
        self.cancer_test_rmse = self._calculate_rmse_for_data(self.cancer_test_data)
        self.healthy_test_rmse = self._calculate_rmse_for_data(self.healthy_test_data)
        self.healthy_train_rmse = self._calculate_rmse_for_data(self.healthy_train_data)

    def _calculate_rmse_for_data(self, data):
        predictions = self.model.predict(data)
        rmse = np.sqrt(np.mean((predictions - data) ** 2, axis=1))
        return rmse

    def display_average_rmse(self):
        print(f"Healthy train RMSE: {np.mean(self.healthy_train_rmse)}")
        print(f"Healthy test RMSE: {np.mean(self.healthy_test_rmse)}")
        print(f"Cancer train RMSE: {np.mean(self.cancer_train_rmse)}")
        print(f"Cancer test RMSE: {np.mean(self.cancer_test_rmse)}")
        print(f"Threshold: {self.threshold}")

    def _calculate_threshold(self):
        self._calculate_rmse_list()
        cancer_train_mean = np.mean(self.cancer_train_rmse)
        healthy_train_mean = np.mean(self.healthy_train_rmse)
        return (cancer_train_mean + healthy_train_mean) / 2

    def predict(self, data):
        rmse = self._calculate_rmse_for_data(data)
        results = np.zeros(len(data))
        for i in range(len(data)):
            if rmse[i] >= self.threshold:
                results[i] = 1
        return results

    def display_regular_metrics(self, data, classes, set_name):
        predictions = self.predict(data)

        balanced_accuracy = balanced_accuracy_score(classes, predictions)
        print(f"{set_name} balanced accuracy: {balanced_accuracy}")

        f1 = f1_score(classes, predictions)
        print(f"{set_name} f1 score: {f1}")

        precision = precision_score(classes, predictions)
        print(f"{set_name} precision: {precision}")

        recall = recall_score(classes, predictions)
        print(f"{set_name} recall: {recall}")
