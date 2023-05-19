import numpy as np
from scipy.stats import ttest_ind, shapiro, levene, ranksums
from statsmodels.stats.multitest import fdrcorrection
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif


def draw_dendrogram(data, linkage_method, p):
    plt.figure(figsize=(15, 8))
    dendro = sch.dendrogram(
        sch.linkage(data, method=linkage_method), truncate_mode="lastp", p=p
    )
    plt.ylabel(f"kryterium łączenia klastrów ({linkage_method})", fontsize=15)
    plt.title(f"dendrogram dla kryterium: {linkage_method}", fontsize=15)
    plt.show


def draw_heatmap(data):
    sns.clustermap(data)
    plt.show()


def calculate_statistics(data, true_results):
    cancer_cells_indices = np.where(true_results == 1)[0]
    healthy_cells_indices = np.where(true_results == 0)[0]

    healthy_cells_data = data.iloc[healthy_cells_indices]
    cancer_cells_data = data.iloc[cancer_cells_indices]

    statistics = pd.DataFrame()

    statistics["ctc_mean"] = np.mean(cancer_cells_data, axis=0)
    statistics["other_cells_mean"] = np.mean(healthy_cells_data, axis=0)

    statistics["ctc_std"] = np.std(cancer_cells_data)
    statistics["other_cells_std"] = np.std(healthy_cells_data)

    levene_test = calculate_levene_test(cancer_cells_data, healthy_cells_data)
    shapiro_test = calculate_shapiro_test(cancer_cells_data, healthy_cells_data)

    for column in data.columns:
        if (
            shapiro_test.at[column, "cancer-cells-p-values"] > 0.05
            and shapiro_test.at[column, "healthy-cells-p-values"] > 0.05
        ):
            is_variance_equal = levene_test.at[column, "p-values"] > 0.05

            _, p_value = ttest_ind(
                cancer_cells_data[column],
                healthy_cells_data[column],
                equal_var=is_variance_equal,
            )
            statistics.at[column, "p-values"] = p_value

        else:
            _, p_value = ranksums(cancer_cells_data[column], healthy_cells_data[column])
            statistics.at[column, "p-values"] = p_value

    return statistics.sort_values(by="p-values")


def calculate_global_statistics(data, true_results):
    cancer_cells_indices = np.where(true_results == 1)[0]
    healthy_cells_indices = np.where(true_results == 0)[0]

    healthy_cells_data = data.iloc[healthy_cells_indices]
    cancer_cells_data = data.iloc[cancer_cells_indices]

    print(f"CTC mean: {np.mean(cancer_cells_data.to_numpy())}")
    print(f"WBC mean: {np.mean(healthy_cells_data.to_numpy())}")

    print(f"CTC standard deviation: {np.std(cancer_cells_data.to_numpy())}")
    print(f"WBC standard deviation: {np.std(healthy_cells_data.to_numpy())}")

    print(f"CTC non-zero values: {get_non_zero_values_percent(cancer_cells_data)} %")
    print(f"WBC non-zero values: {get_non_zero_values_percent(healthy_cells_data)} %")

    test = cancer_cells_data.to_numpy().flatten()

    _, levene_p_value = levene(
        healthy_cells_data.to_numpy().flatten().T,
        cancer_cells_data.to_numpy().flatten().T,
    )

    _, healthy_cells_shapiro_p_value = shapiro(
        healthy_cells_data.to_numpy().flatten().T
    )
    _, cancer_cells_shapiro_p_value = shapiro(cancer_cells_data.to_numpy().flatten().T)

    if healthy_cells_shapiro_p_value > 0.05 and cancer_cells_shapiro_p_value > 0.05:
        is_variance_equal = levene_p_value > 0.05

        _, p_value = ttest_ind(
            cancer_cells_data.to_numpy().flatten().T,
            healthy_cells_data.to_numpy().flatten().T,
            equal_var=is_variance_equal,
        )

    else:
        _, p_value = ranksums(
            cancer_cells_data.to_numpy().flatten().T,
            healthy_cells_data.to_numpy().flatten().T,
        )

    print(f"T-test p-values: {p_value}")


def display_neural_network_metrics(
    model, batch_size_value, data, true_results, set_name
):
    results = model.evaluate(data, true_results, batch_size=batch_size_value)

    print(f"{set_name} loss: {results[0]}")
    print(f"{set_name} binary accuracy: {results[1]}")
    print(f"{set_name} auc: {results[2]}")
    print(f"{set_name} precision: {results[3]}")
    print(f"{set_name} recall: {results[4]}")


def get_best_shap_features(shap_results, feature_names, amount):
    rf_result_x = pd.DataFrame(shap_results[0], columns=feature_names)
    vals = np.abs(rf_result_x.values).mean(0)
    shap_importance = pd.DataFrame(
        list(zip(feature_names, vals)), columns=["col_name", "feature_importance_vals"]
    )
    shap_importance.sort_values(
        by=["feature_importance_vals"], ascending=False, inplace=True
    )
    best_features = list(shap_importance.head(amount)["col_name"])
    return best_features


def get_best_metrics(clfs_names, metric):
    return [
        clfs_names[x] for x in [idx for idx, x in enumerate(metric) if x == max(metric)]
    ]


def variance_between(d1, d2):
    d = pd.concat([d1, d2], axis=0)

    p1 = len(d1) / len(d)
    p2 = len(d2) / len(d)

    var = (p1 * p2) * pow((d1.mean() - d2.mean()), 2)

    return var


def calculate_levene_test(cancer_cells_data, healthy_cells_data):
    p_values = pd.DataFrame(index=healthy_cells_data.columns)
    for column in healthy_cells_data.columns:
        _, p_value = levene(healthy_cells_data[column], cancer_cells_data[column])
        p_values.at[column, "p-values"] = p_value
    return p_values


def calculate_shapiro_test(cancer_cells_data, healthy_cells_data):
    shapiro_test = pd.DataFrame(index=healthy_cells_data.columns)
    for column in healthy_cells_data.columns:
        _, healthy_cells_p_value = shapiro(healthy_cells_data[column])
        _, cancer_cells_p_value = shapiro(cancer_cells_data[column])
        shapiro_test.at[column, "cancer-cells-p-values"] = cancer_cells_p_value
        shapiro_test.at[column, "healthy-cells-p-values"] = healthy_cells_p_value
    return shapiro_test


def get_non_zero_values_percent(data):
    count = np.count_nonzero(data)
    total_values = len(data.index.values) * len(data.columns.values)
    percent_value = (count * 100) / total_values
    return percent_value


def get_non_zero_feature_importance_of_tree_base_classifier(models, data):
    feature_importance_array = []
    for model in models:
        model_feature_importance = model.feature_importances_
        feature_importance_array.append(model_feature_importance)
    average_feature_importance = np.mean(feature_importance_array, axis=0)

    feature_importance = pd.DataFrame(
        index=data.columns, data=average_feature_importance
    )
    final_feature_importance = feature_importance.loc[
        (feature_importance != 0).any(axis=1)
    ]
    final_features = final_feature_importance.index.values
    return final_features, final_feature_importance
