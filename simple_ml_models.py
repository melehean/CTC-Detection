from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np
import pandas as pd

from utils import get_best_metrics


class Model:
    def __init__(
        self,
        model,
        train_data,
        train_true_results,
        test_data,
        test_true_results,
        fold_number,
        minus_one_one_values=False,
    ):
        self.model = model
        self.train_data = train_data
        self.fold_number = fold_number
        self.test_data = test_data
        self.minus_one_one_values = minus_one_one_values

        if minus_one_one_values:
            self.test_true_results = test_true_results.replace([0, 1], [1, -1])
            self.train_true_results = train_true_results.replace([0, 1], [1, -1])
        else:
            self.test_true_results = test_true_results
            self.train_true_results = train_true_results

        self.scoring = ["balanced_accuracy", "roc_auc", "precision", "recall", "f1"]
        self.test_balanced_accuracy_list = []
        self.test_roc_auc_list = []
        self.test_f1_score_list = []
        self.test_recall_list = []
        self.test_precision_list = []

    def main_cycle(self):
        cv_results = cross_validate(
            self.model,
            self.train_data,
            self.train_true_results,
            cv=self.fold_number,
            return_train_score=True,
            scoring=self.scoring,
            return_estimator=True,
        )
        # Model.display_metrics(cv_results)

        for cross_validation_model in cv_results["estimator"]:
            self.gather_test_results(cross_validation_model)

        return cv_results["estimator"]

    def gather_test_results(self, model):
        predictions = model.predict(self.test_data)
        predictions = predictions.flatten()

        test_balanced_accuracy = balanced_accuracy_score(
            self.test_true_results, predictions
        )
        self.test_balanced_accuracy_list.append(test_balanced_accuracy)

        if self.minus_one_one_values:
            scores = model.decision_function(self.test_data)
            test_roc_auc = roc_auc_score(self.test_true_results, scores)
            self.test_roc_auc_list.append(test_roc_auc)
        else:
            test_roc_auc = roc_auc_score(
                self.test_true_results, model.predict_proba(self.test_data)[::, 1]
            )
            self.test_roc_auc_list.append(test_roc_auc)

        test_precision = precision_score(self.test_true_results, predictions)
        self.test_precision_list.append(test_precision)

        test_recall = recall_score(self.test_true_results, predictions)
        self.test_recall_list.append(test_recall)

        test_f1_score = f1_score(self.test_true_results, predictions)
        self.test_f1_score_list.append(test_f1_score)

    def display_test_results(self):
        print("Test balanced accuracy", np.mean(self.test_balanced_accuracy_list))
        print()

        print("Test roc auc", np.mean(self.test_roc_auc_list))
        print()

        print("Test precision", np.mean(self.test_precision_list))
        print()

        print("Test recall", np.mean(self.test_recall_list))
        print()

        print("Test f1 score", np.mean(self.test_f1_score_list))

    @staticmethod
    def display_metrics(cv_results):
        for value in cv_results.keys():
            str_value = str(value)
            if str_value.startswith("test_"):
                str_value = str_value.replace("test", "Validation")
                str_value = str_value.replace("_", " ")
                print(str_value, cv_results[value].mean())
            if str_value.startswith("train_"):
                str_value = str_value.replace("train", "Training")
                str_value = str_value.replace("_", " ")
                print(str_value, cv_results[value].mean())
                print()


def run_all_models(
    classifiers,
    clfs_names,
    train_data,
    train_true_results,
    test_data,
    test_true_results,
):
    balanced_acc = []
    roc_auc = []
    precision = []
    recall = []
    f1 = []

    for (clf_name, clf) in zip(clfs_names, classifiers):
        minus_one_one_values = False

        if clf_name == "Isolated Forest":
            minus_one_one_values = True

        model_object = Model(
            model=clf,
            train_data=train_data,
            train_true_results=train_true_results,
            test_data=test_data,
            test_true_results=test_true_results,
            fold_number=3,
            minus_one_one_values=minus_one_one_values,
        )
        model_object.main_cycle()

        balanced_acc.append(np.mean(model_object.test_balanced_accuracy_list))
        roc_auc.append(np.mean(model_object.test_roc_auc_list))
        precision.append(np.mean(model_object.test_precision_list))
        recall.append(np.mean(model_object.test_recall_list))
        f1.append(np.mean(model_object.test_f1_score_list))

    data = {
        "Balanced Accuracy": balanced_acc,
        "ROC AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1,
    }
    results = pd.DataFrame(data=data, index=clfs_names)

    print(f"Features number: {len(train_data.columns)}")
    print(
        "Best balanced accuracy:", ", ".join(get_best_metrics(clfs_names, balanced_acc))
    )
    print("Best ROC AUC:", ", ".join(get_best_metrics(clfs_names, roc_auc)))
    print("Best precision:", ", ".join(get_best_metrics(clfs_names, precision)))
    print("Best recall:", ", ".join(get_best_metrics(clfs_names, recall)))
    print("Best F1 score:", ", ".join(get_best_metrics(clfs_names, f1)))

    return results
