import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns


class Visualisation:
    def __init__(self, models, data, classes, classes_names=None):
        self.models = models
        self.data = data
        self.classes = classes
        self.classes_names = classes_names

        self.average_healthy_cells_probabilities = None
        self.average_cancer_cells_probabilities = None
        self.average_mix_cells_probabilites = None
        self.average_all_cells_probabilites = None

        self.mix_boxplot = False

        if classes_names is not None:
            self.gather_probabilites_by_name()
            self.mix_boxplot = True
        else:
            self.gather_probabilites_by_number()

    def gather_probabilites_by_name(self):
        healthy_cells_probabilities = []
        cancer_cells_probabilities = []
        mix_cells_probabilites = []
        all_cells_probabilites = []

        healthy_cells_test_indices = np.where(self.classes_names == "WBC")[0]
        healthy_test_data = self.data.iloc[healthy_cells_test_indices]

        cancer_cells_test_indices = np.where(self.classes_names == "CTC")[0]
        cancer_test_data = self.data.iloc[cancer_cells_test_indices]

        mix_cells_test_indices = np.where(self.classes_names == "CTC-WBC")[0]
        mix_test_data = self.data.iloc[mix_cells_test_indices]

        for model in self.models:
            healthy_cells_proba = model.predict_proba(healthy_test_data)[::, 1]
            cancer_cells_proba = model.predict_proba(cancer_test_data)[::, 1]
            mix_cells_proba = model.predict_proba(mix_test_data)[::, 1]
            all_cells_proba = model.predict_proba(self.data)[::, 1]

            healthy_cells_probabilities.append(healthy_cells_proba)
            cancer_cells_probabilities.append(cancer_cells_proba)
            mix_cells_probabilites.append(mix_cells_proba)
            all_cells_probabilites.append(all_cells_proba)

        self.average_healthy_cells_probabilities = np.mean(
            healthy_cells_probabilities, axis=0
        )
        self.average_cancer_cells_probabilities = np.mean(
            cancer_cells_probabilities, axis=0
        )
        self.average_mix_cells_probabilites = np.mean(mix_cells_probabilites, axis=0)
        self.average_all_cells_probabilites = np.mean(all_cells_probabilites, axis=0)

    def gather_probabilites_by_number(self):
        healthy_cells_probabilities = []
        cancer_cells_probabilities = []
        all_cells_probabilites = []

        healthy_cells_test_indices = np.where(self.classes == 0)[0]
        healthy_test_data = self.data.iloc[healthy_cells_test_indices]

        cancer_cells_test_indices = np.where(self.classes == 1)[0]
        cancer_test_data = self.data.iloc[cancer_cells_test_indices]

        for model in self.models:
            healthy_cells_proba = model.predict_proba(healthy_test_data)[::, 1]
            cancer_cells_proba = model.predict_proba(cancer_test_data)[::, 1]
            all_cells_proba = model.predict_proba(self.data)[::, 1]

            healthy_cells_probabilities.append(healthy_cells_proba)
            cancer_cells_probabilities.append(cancer_cells_proba)
            all_cells_probabilites.append(all_cells_proba)

        self.average_healthy_cells_probabilities = np.mean(
            healthy_cells_probabilities, axis=0
        )
        self.average_cancer_cells_probabilities = np.mean(
            cancer_cells_probabilities, axis=0
        )
        self.average_all_cells_probabilites = np.mean(all_cells_probabilites, axis=0)

    def plot_predictions_boxplot(
        self, cancer_title=None, healthy_title=None, mixed_title=None
    ):
        if self.mix_boxplot:
            fig, ax = plt.subplots(1, 3)
            fig.set_figheight(4)
            fig.set_figwidth(12)
            ax[0].boxplot(self.average_cancer_cells_probabilities)
            ax[0].set_title(cancer_title)
            ax[0].set_ylim(-0.05, 1.05)
            ax[1].boxplot(self.average_healthy_cells_probabilities)
            ax[1].set_title(healthy_title)
            ax[1].set_ylim(-0.05, 1.05)
            ax[2].boxplot(self.average_mix_cells_probabilites)
            ax[2].set_title(mixed_title)
            ax[2].set_ylim(-0.05, 1.05)
            plt.show()
        else:
            fig, ax = plt.subplots(1, 2)
            fig.set_figheight(4)
            fig.set_figwidth(12)
            ax[0].boxplot(self.average_cancer_cells_probabilities)
            ax[0].set_title(cancer_title)
            ax[0].set_ylim(-0.05, 1.05)
            ax[1].boxplot(self.average_healthy_cells_probabilities)
            ax[1].set_title(healthy_title)
            ax[1].set_ylim(-0.05, 1.05)
            plt.show()

    def draw_roc_curve_from_probabilities(self):
        fpr, tpr, _ = roc_curve(self.classes, self.average_all_cells_probabilites)
        auc = roc_auc_score(self.classes, self.average_all_cells_probabilites)
        plt.plot(fpr, tpr, label="AUC=" + str(round(auc, 2)))
        plt.legend(loc=4)
        plt.show()

    def display_confusion_matrix(self, cancer_category, healthy_category):
        predictions = np.round(self.average_all_cells_probabilites)
        cf_matrix = confusion_matrix(self.classes, predictions)
        categories = [healthy_category, cancer_category]

        group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
        group_percentages = [
            "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
        ]
        labels = [f"{v1} ({v2})" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(
            cf_matrix,
            annot=labels,
            fmt="",
            cmap="Blues",
            xticklabels=categories,
            yticklabels=categories,
        )
