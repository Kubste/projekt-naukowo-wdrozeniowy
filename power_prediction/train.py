import os
import json
import numpy as np
from utils.paths import DATASETS_PREPROCESSED_DIR
from joblib import dump
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error


with open("power_prediction/results/best_params_all.json", "r") as f:
    all_best_params = json.load(f)


models = {
    "decision_tree":  DecisionTreeRegressor(**all_best_params["decision_tree"]),
    "random_forest":  RandomForestRegressor(**all_best_params["random_forest"]),
    "svr":            SVR(**all_best_params["svr"]),
    "catboost":       CatBoostRegressor(**all_best_params["catboost"]),
    "xgboost":        XGBRegressor(**all_best_params["xgboost"]),
    "lgbm":           LGBMRegressor(**all_best_params["lgbm"]),
    "mlp":            MLPRegressor(**all_best_params["mlp"], early_stopping=True)
    }

skf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=44)
mean_scores = pd.DataFrame(columns=["model", "MSE", "MAE", "R^2", "MAPE"])

for i in range(1, 5):      # load train, test data
    data_train = pd.read_csv(os.path.join(DATASETS_PREPROCESSED_DIR,f"Location_{i}_train.csv"))
    data_test = pd.read_csv(os.path.join(DATASETS_PREPROCESSED_DIR,f"Location_{i}_test.csv"))
    data = pd.concat([data_train, data_test])
    targets = data["delta_p"]
    data = data.drop(columns=["time", "p_base", "delta_p"])  
    mean_scores = pd.DataFrame(columns=["model", "MSE", "MAE", "R^2", "MAPE"])
    for name, model in models.items():
        scores = pd.DataFrame(columns=["MSE", "MAE", "R^2", "MAPE"])
        mse = 100
        mae = 100
        r2 = -10
        mape = 100
        for j, (train, test) in enumerate(skf.split(data, targets)):
            cnt = 0
            X_train = data.iloc[train]
            y_train = targets.iloc[train]
            X_test = data.iloc[test]
            y_test = targets.iloc[test]
            # train model
            model = clone(model)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # calculate metrics
            cur_mse = mean_squared_error(y_test, y_pred) 
            cur_mae = mean_absolute_error(y_test, y_pred)
            cur_r2 = r2_score(y_test, y_pred)
            cur_mape = mean_absolute_percentage_error(y_test, y_pred)
            scores.loc[j] = [cur_mse, cur_mae, cur_r2, cur_mape]
            # check if model has better metric than current best
            if cur_mse < mse:
                cnt += 1
            if cur_mae < mae:
                cnt += 1
            if cur_r2 > r2:
                cnt += 1
            # save if it is better in more than 2 metrics than current best
            if cnt > 1:
                with open(rf"power_prediction\models\{name}_loc{i}.pkl", 'wb') as fo:
                    dump(model, fo)


        print(f"{name}{i} done")
        # dataframe with metrics scores (each row is one split)
        scores.to_csv(rf"power_prediction\results\location{i}\{name}_scores.csv")
        mean_score = pd.Series({"model": name, "MSE": scores["MSE"].mean(), "MAE": scores["MAE"].mean(), "R^2": scores["R^2"].mean(), "MAPE": scores["MAPE"].mean()})
        mean_scores = pd.concat([mean_scores, mean_score.to_frame().T], ignore_index=True)

    mean_scores.to_csv(rf"power_prediction\results\location{i}\avg_scores.csv")



