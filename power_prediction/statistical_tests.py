from scipy.stats import wilcoxon
import pandas as pd
from itertools import combinations

models = [
    "decision_tree",
    "random_forest",
    "svr",
    "catboost",
    "xgboost",
    "lgbm",
    "mlp",
]

for i in range(1,5):
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

    test_mse = pd.DataFrame(columns={"model1", "model2", "p_value"})
    for i, x in enumerate(mse):
        if i == 9:
            break
        for j in range(i, 10):
            y = mse[j]
            _, p_value = wilcoxon(x, y)
            test_mse = pd.concat([test_mse, pd.Series({"model1": models[i], "model2": models[j], "p_value": p_value}).to_frame().T], ignore_index=True)

    test_mae = pd.DataFrame(columns={"model1", "model2", "p_value"})
    for i, x in enumerate(mae):
        if i == 9:
            break
        for j in range(i, 10):
            y = mae[j]
            _, p_value = wilcoxon(x, y)
            test_mae = pd.concat([test_mae, pd.Series({"model1": models[i], "model2": models[j], "p_value": p_value}).to_frame().T], ignore_index=True)


    test_r2 = pd.DataFrame(columns={"model1", "model2", "p_value"})
    for i, x in enumerate(r2):
        if i == 9:
            break
        for j in range(i, 10):
            y = r2[j]
            _, p_value = wilcoxon(x, y)
            test_r2 = pd.concat([test_r2, pd.Series({"model1": models[i], "model2": models[j], "p_value": p_value}).to_frame().T], ignore_index=True)

    test_mape = pd.DataFrame(columns={"model1", "model2", "p_value"})
    for i, x in enumerate(mape):
        if i == 9:
            break
        for j in range(i, 10):
            y = mape[j]
            _, p_value = wilcoxon(x, y)
            test_mape = pd.concat([test_mape, pd.Series({"model1": models[i], "model2": models[j], "p_value": p_value}).to_frame().T], ignore_index=True)

    test_mae.to_csv(rf"power_prediction\results\location{i}\wilcoxon_results.csv")
    test_mse.to_csv(rf"power_prediction\results\location{i}\wilcoxon_results.csv")
    test_r2.to_csv(rf"power_prediction\results\location{i}\wilcoxon_results.csv")
    test_mape.to_csv(rf"power_prediction\results\location{i}\wilcoxon_results.csv")


