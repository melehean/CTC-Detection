from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class FeatureImportance:
    REGULARIZATION = 0.02
    IMPORTANCE_THRESHOLD = 0.02

    def __init__(self, train_data, test_data, train_true_results, test_true_results, columns):
        self.train_data = train_data
        self.test_data = test_data
        self.train_true_results = train_true_results
        self.test_true_results = test_true_results
        self.columns = columns

    def logistic_regression_coefficients_importance(self):
        clf = LogisticRegression(random_state=0, class_weight='balanced', penalty='l1', C=self.REGULARIZATION,
                                 solver='liblinear')
        clf.fit(self.train_data, self.train_true_results)
        print("Train accuracy: " + str(clf.score(self.train_data, self.train_true_results)))
        print("Test accuracy: " + str(clf.score(self.test_data, self.test_true_results)))
        zipped = sorted(zip(clf.coef_.tolist()[0], self.columns))
        genes_that_matter = list(filter(lambda x: abs(x[0]) > self.IMPORTANCE_THRESHOLD, zipped))
        print("Number of genes that matter: " + str(len(genes_that_matter)) + " out of " + str(len(zipped)) +
              "  (" + str(100 * len(genes_that_matter) / len(zipped)) + "%)")
        genes_importances = list(zip(*genes_that_matter))
        plt.plot(genes_importances[1], genes_importances[0])
        plt.title("Importances of genes that matter in prediction (" + str(100 * len(genes_that_matter) / len(
            zipped)) + "% of all genes)" + "\nPositive values mean genes, which predict cancer,\nnegative values are genes, which predict absence of cancer")
        plt.xticks(rotation=90)
        plt.tick_params(axis='x', which='major', labelsize=6)
        plt.show()
