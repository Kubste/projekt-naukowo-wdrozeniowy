import os
import json
import numpy as np
from utils.paths import DATASETS_PREPROCESSED_DIR
from joblib import dump, load
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from scipy.stats import wilcoxon

with open(r"power_prediction\results\best_params_all.json", "r") as f:
    all_best_params = json.load(f)

models = [
    "decision_tree",
    "random_forest",
    "svr",
    "catboost",
    "xgboost",
    "mlp"  ]
base_learners = []

for i in range(4):
    base_learners.append([])
    for name in models:
        with open(rf"power_prediction\models\{name}_loc{i+1}.pkl", "rb") as f:
            model = load(f)
            base_learners[i].append((name, model))

skf = RepeatedKFold(n_repeats=5, n_splits=2, random_state=44)

meta_model = XGBRegressor(**all_best_params["xgboost"])
metrics = ["MSE", "MAE", "R^2", "MAPE"]
scores = pd.DataFrame(columns=metrics)

# dataframe for wilcoxon test results
test_results = pd.DataFrame(columns=metrics)

for i in range(1,5):
    scores = pd.DataFrame(columns=["MSE", "MAE", "R^2", "MAPE"])
    stacking_model = StackingRegressor(estimators=base_learners[i-1], final_estimator=meta_model, n_jobs=12)
    data_train = pd.read_csv(os.path.join(DATASETS_PREPROCESSED_DIR,f"Location_{i}_train.csv"))
    data_test = pd.read_csv(os.path.join(DATASETS_PREPROCESSED_DIR,f"Location_{i}_test.csv"))
    data = pd.concat([data_train, data_test])
    targets = data["delta_p"]
    data = data.drop(columns=["time", "p_base", "delta_p"])  
    mse = 100
    mae = 100
    r2 = -10
    cnt = 0
    for j, (train, test) in enumerate(skf.split(data, targets)):
        X_train = data.iloc[train]
        y_train = targets.iloc[train]
        X_test = data.iloc[test]
        y_test = targets.iloc[test]
        stacking_model.fit(X_train, y_train)
        y_pred = stacking_model.predict(X_test)
        # calculate metrics
        cur_mse = mean_squared_error(y_test, y_pred) 
        cur_mae = mean_absolute_error(y_test, y_pred)
        cur_r2 = r2_score(y_test, y_pred)
        cur_mape = mean_absolute_percentage_error(y_test, y_pred)
        scores.loc[j-1] = [cur_mse, cur_mae, cur_r2, cur_mape]
        # check if model has better metric than current best
        if cur_mse < mse:
            cnt += 1
        if cur_mae < mae:
            cnt += 1
        if cur_r2 > r2:
            cnt += 1
        # save if it is better in more than 2 metrics than current best
        if cnt > 1:
            with open(rf"power_prediction\models\stacking_model_loc{i}.pkl", 'wb') as fo:
                dump(stacking_model, fo)

    print(f"{i} location done")

    # dataframe with metrics scores (each row is one split)
    scores.to_csv(rf"power_prediction\results\location{i}\stacking_model_scores.csv")

    # wilcoxon test to compare stacking model vs best standalone model
    xgb_scores = pd.read_csv(rf"power_prediction\results\location{i}\xgboost_scores.csv")
    p_values_list = []
    for metric in metrics:
        _, p_value = wilcoxon(xgb_scores[metric], scores[metric])
        p_values_list.append(p_value)
                
    test_results.loc[i-1] = p_values_list



test_results.to_csv(rf"power_prediction\results\wicoxon_results_stacking_vs_best.csv")
