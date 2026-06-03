from scipy.stats import wilcoxon
import pandas as pd

models = [
    "decision_tree",
    "random_forest",
    "svr",
    "catboost",
    "xgboost",
    "lgbm",
    "mlp",
]

def sort_df_by_metric(df, mean_scores, metric="MSE", ascending=True):

    # Get the sorted model order from mean_scores
    sorted_models = (
        mean_scores
        .sort_values(by=metric, ascending=ascending)["model"]
        .tolist()
    )

    # Reindex both axes
    return df.loc[sorted_models, sorted_models]



# test for each location and each metric
for i in range(1,5):  
    mean_scores = pd.read_csv(rf"power_prediction\results\location{i}\avg_scores.csv")
    mse = []
    mae = []
    mape = []
    r2 = []
    for name in models:
        scores = pd.read_csv(rf"power_prediction\results\{name}_loc{i}_scores.csv")
        mse.append(scores["MSE"])
        mae.append(scores["MAE"])
        r2.append(scores["R^2"])
        mape.append(scores["MAPE"])

    test_mse = pd.DataFrame(columns=models, index=models)
    sort_df_by_metric(test_mse, mean_scores, metric="MSE") = test_mse
    for i, x in enumerate(mse):
        if i == 9:
            break
        for j in range(i, 10):
            y = mse[j]
            _, p_value = wilcoxon(x, y)
            test_mse.at[models[i], models[j]] = p_value
            test_mse.at[models[j], models[i]] = p_value

    test_mae = pd.DataFrame(columns=models, index=models)
    sort_df_by_metric(test_mae, mean_scores, metric="MAE") = test_mae
    for i, x in enumerate(mae):
        if i == 9:
            break
        for j in range(i, 10):
            y = mae[j]
            _, p_value = wilcoxon(x, y)
            test_mae.at[models[i], models[j]] = p_value
            test_mae.at[models[j], models[i]] = p_value

    test_r2 = pd.DataFrame(columns=models, index=models)
    sort_df_by_metric(test_r2, mean_scores, metric="R^2", ascending=False) = test_r2
    for i, x in enumerate(r2):
        if i == 9:
            break
        for j in range(i, 10):
            y = r2[j]
            _, p_value = wilcoxon(x, y)
            test_r2.at[models[i], models[j]] = p_value
            test_r2.at[models[j], models[i]] = p_value
    test_mape = pd.DataFrame(columns=models, index=models)
    sort_df_by_metric(test_mape, mean_scores, metric="MAPE") = test_mape
    for i, x in enumerate(mape):
        if i == 9:
            break
        for j in range(i, 10):
            y = mape[j]
            _, p_value = wilcoxon(x, y)
            test_mae.at[models[i], models[j]] = p_value
            test_mae.at[models[j], models[i]] = p_value

    test_mae.to_csv(rf"power_prediction\results\location{i}\wilcoxon_results.csv")
    test_mse.to_csv(rf"power_prediction\results\location{i}\wilcoxon_results.csv")
    test_r2.to_csv(rf"power_prediction\results\location{i}\wilcoxon_results.csv")
    test_mape.to_csv(rf"power_prediction\results\location{i}\wilcoxon_results.csv")


