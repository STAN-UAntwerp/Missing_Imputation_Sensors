from utils.evaluation import calculate_metrics_random_missings, calculate_metrics_real_missings, calculate_all_metrics, simple_eval, plot_metric_percentages

if __name__ == '__main__':

    dataset = 'CN'
    methods = ['mean_imputation', 'spline_imputation', 'cubic_spline_imputation', 'soft_imputation_scaled', 'soft_imputation_not_scaled', 'knn_imputation',
               'mf_imputation', 'AKE_imputation_k=5']#, 'DESM_imputation']
    # for method in methods:
    #     print(method)
    #     calculate_metrics_random_missings(method, dataset)

    # calculate_metrics_random_missings('AKE_imputation', dataset)

    plot_metric_percentages(methods=methods, metric='rmse', dataset=dataset)
    plot_metric_percentages(methods=methods, metric='mae', dataset=dataset)
    plot_metric_percentages(methods=methods, metric='pce', dataset=dataset)