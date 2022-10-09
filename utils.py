import numpy as np
import scipy.stats as stats
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics


def draw_dendrogram(data, linkage_method, p):
    plt.figure(figsize=(15,8))
    dendro = sch.dendrogram(sch.linkage(data, method=linkage_method), truncate_mode='lastp', p=p)
    plt.ylabel(f'kryterium łączenia klastrów ({linkage_method})',fontsize=15)
    plt.title(f'dendrogram dla kryterium: {linkage_method}',fontsize=15)
    plt.show
    

def draw_heatmap(data):
    sns.clustermap(data)
    plt.show()


def calculate_statistics(data, true_results):
    cancer_cells_indices = np.where(true_results == 1)[0]
    healthy_cells_indices = np.where(true_results == 0)[0]

    healthy_cells_data = data.iloc[healthy_cells_indices]
    cancer_cells_data = data.iloc[cancer_cells_indices]

    d = {'ctc_variance': np.var(cancer_cells_data), 'other_cells_variance': np.var(healthy_cells_data)}
    statistics = pd.DataFrame(data=d)

    statistics["ctc_mean"] = np.mean(cancer_cells_data, axis=0)
    statistics["other_cells_mean"] = np.mean(healthy_cells_data, axis=0)

    statistics["ctc_std"] = np.std(cancer_cells_data)
    statistics["other_cells_std"] = np.std(healthy_cells_data)

    variance_check_1 = statistics['ctc_variance']/statistics['other_cells_variance'] < 4
    variance_check_2 = statistics['other_cells_variance']/statistics['ctc_variance'] < 4
    statistics["variance_check"] = variance_check_1 & variance_check_2

    t_statistics = np.where(statistics["variance_check"], stats.ttest_ind(healthy_cells_data, cancer_cells_data, equal_var = True).statistic, stats.ttest_ind(healthy_cells_data, cancer_cells_data, equal_var = False).statistic)
    t_statistics = np.absolute(t_statistics)
    statistics['t_test'] = t_statistics

    p_values = np.where(statistics["variance_check"], stats.ttest_ind(healthy_cells_data, cancer_cells_data, equal_var = True).pvalue, stats.ttest_ind(healthy_cells_data, cancer_cells_data, equal_var = False).pvalue)
    statistics['p_values'] = p_values

    return statistics.sort_values(by="p_values")


def display_autoencoder_metrics(model, train_data, test_data, cancer_data):
    test_predictions = model.predict(test_data)
    train_predictions = model.predict(train_data)
    cancer_data_predictions = model.predict(cancer_data)

    test_score = np.sqrt(metrics.mean_squared_error(test_predictions, test_data))
    train_score = np.sqrt(metrics.mean_squared_error(train_predictions, train_data))
    cancer_data_score = np.sqrt(metrics.mean_squared_error(cancer_data_predictions, cancer_data))

    print(f"Train RMSE: {train_score}")
    print(f"Test RMSE: {test_score}")
    print(f"Cancer data RMSE: {cancer_data_score}")
