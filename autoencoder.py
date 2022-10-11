import numpy as np
from sklearn import metrics


class Autoencoder:
    def __init__(self, model, train_data, test_data, cancer_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.cancer_data = cancer_data
        self.cancer_rmse = None
        self.test_rmse = None
        self.train_rmse = None
        self._calculate_rmse_list()

    def _calculate_rmse_list(self):
        test_predictions = self.model.predict(self.test_data)
        train_predictions = self.model.predict(self.train_data)
        cancer_predictions = self.model.predict(self.cancer_data)

        self.cancer_rmse = np.sqrt(np.mean((cancer_predictions - self.cancer_data) ** 2, axis=1))
        self.test_rmse = np.sqrt(np.mean((test_predictions - self.test_data) ** 2, axis=1))
        self.train_rmse = np.sqrt(np.mean((train_predictions - self.train_data) ** 2, axis=1))

    def display_average_rmse(self):
        print(f"Train RMSE: {np.mean(self.train_rmse)}")
        print(f"Test RMSE: {np.mean(self.test_rmse)}")
        print(f"Cancer data RMSE: {np.mean(self.cancer_rmse)}")

    def calculate_threshold(self):
        cancer_mean = np.mean(self.cancer_rmse)
        test_mean = np.mean(self.test_rmse)
        train_mean = np.mean(self.train_rmse)
        healthy_max = max(train_mean, test_mean)

        return (cancer_mean + healthy_max) / 2

    def _get_classification_results(self):
        threshold = self.calculate_threshold()
        cancer_results = self.cancer_rmse >= threshold
        test_results = self.test_rmse < threshold
        train_results = self.train_rmse < threshold
        return train_results, test_results, cancer_results

    @staticmethod
    def _display_set_metrics(results, set_name):
        accuracy = len(results[results == 1]) / len(results)
        print(f"{set_name} accuracy: {accuracy}")

    def display_classification_metrics(self):
        train_results, test_results, cancer_results = self._get_classification_results()
        self._display_set_metrics(train_results, "Healthy Train")
        self._display_set_metrics(test_results, "Healthy Test")
        self._display_set_metrics(cancer_results, "Cancer")
