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


def display_neural_network_metrics(model, batch_size_value, data, true_results, set_name):
    results = model.evaluate(data, true_results, batch_size=batch_size_value)

    print(f"{set_name} loss: {results[0]}")
    print(f"{set_name} binary accuracy: {results[1]}")
    print(f"{set_name} auc: {results[2]}")
    print(f"{set_name} precision: {results[3]}")
    print(f"{set_name} recall: {results[4]}")
