import os
import json
import numpy as np
from utils.paths import DATASETS_PREPROCESSED_DIR
from joblib import dump, load
import pandas as pd
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error

with open("best_params_all.json", "r") as f:
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
        with open(rf"power_prediction\models\{name}{i+1}.pkl", "rb") as f:
            model = load(f)
            base_learners[i].append((name, model))


meta_model = LGBMRegressor(**all_best_params["lgbm"])
scores = pd.DataFrame(columns=["MSE", "MAE", "R^2", "MAPE"])

for i in range(1,5):
    stacking_model = StackingRegressor(estimators=base_learners[i-1], final_estimator=meta_model, n_jobs=12)
    data_train = pd.read_csv(os.path.join(DATASETS_PREPROCESSED_DIR,f"Location_{i}_train.csv"))
    y_train = data_train["delta_p"]
    X_train = data_train.drop(columns=["time", "p_base", "delta_p"])
    data_test = pd.read_csv(os.path.join(DATASETS_PREPROCESSED_DIR,f"Location_{i}_test.csv"))
    y_test = data_test["delta_p"]
    X_test = data_test.drop(columns=["time", "p_base", "delta_p"])
    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)
    # calculate metrics
    mse = mean_squared_error(y_test, y_pred) 
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    scores.loc[i-1] = [mse, mae, r2, mape]

    with open(rf"power_prediction\models\stacking_model{i}.pkl", 'wb') as fo:
        dump(model, fo)
    print(f"{i} done")

# dataframe with metrics scores (each row is one location)
scores.to_csv(rf"power_prediction\scores\stacking_model_scores.csv")

