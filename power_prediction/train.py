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


for name, model in models.items():
    scores = pd.DataFrame(columns=["MSE", "MAE", "R^2", "MAPE"])
    for i in range(1,5):
        # load train, test data
        data_train = pd.read_csv(os.path.join(DATASETS_PREPROCESSED_DIR,f"Location_{i}_train.csv"))
        data_val = pd.read_csv(os.path.join(DATASETS_PREPROCESSED_DIR,f"Location_{i}_val.csv"))
        data_train = pd.concat(data_train, data_val)
        y_train = data_train["delta_p"]
        X_train = data_train.drop(columns=["time", "p_base", "delta_p"])
        data_test = pd.read_csv(os.path.join(DATASETS_PREPROCESSED_DIR,f"Location_{i}_test.csv"))
        y_test = data_test["delta_p"]
        X_test = data_test.drop(columns=["time", "p_base", "delta_p"])
        # train model
        model = clone(model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # calculate metrics
        mse = mean_squared_error(y_test, y_pred) 
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        scores.loc[i-1] = [mse, mae, r2, mape]

        with open(rf"power_prediction\models\{name}{i}.pkl", 'wb') as fo:
            dump(model, fo)
        print(f"{name}{i} done")

    # dataframe with metrics scores (each row is one location)
    scores.to_csv(rf"power_prediction\results\{name}_scores.csv")




