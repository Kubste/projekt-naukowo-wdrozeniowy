import os
import json
from utils.paths import DATASETS_PREPROCESSED_DIR
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv(os.path.join(DATASETS_PREPROCESSED_DIR,"Location_1_val.csv"))
targets = data["delta_p"]
data = data.drop(columns=["time", "p_base", "delta_p"])

all_best_params = {}

parameters = {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'splitter':['best', 'random']}

decision_tree_params = dict(
    max_depth=[None, 3, 5, 10, 20],
    min_samples_split=[2, 5, 10, 20],
    min_samples_leaf=[1, 2, 5, 10],
    max_features=[None, "sqrt", "log2", 0.5],
    criterion=["squared_error", "friedman_mse", "absolute_error"],
    splitter=["best", "random"],
)

random_forest_params = dict(
    n_estimators=[50, 100, 150, 200],
    max_depth=[2, 3, 5, 10],
    min_samples_split=[2, 5, 10],
    min_samples_leaf=[1, 2, 5],
    max_features=["sqrt", "log2", 0.3, 0.5],
    max_samples=[None, 0.7, 0.8, 0.9],  # only when bootstrap=True
)

svr_params = dict(
    C=[0.01, 0.1, 0.5, 1, 2],
    epsilon=[0.01, 0.1, 0.5],
    kernel=["linear", "rbf", "poly", "sigmoid"],
    gamma=["scale", "auto"],
    degree=[2, 3, 4],          # only relevant for poly kernel
    coef0=[0.0, 0.1, 0.5, 1.0],
)

catboost_params = dict(
    iterations=[100, 300, 500, 1000],
    learning_rate=[0.01, 0.05, 0.1],
    depth=[4, 6, 8, 10],
    l2_leaf_reg=[1, 3, 5, 10],
    verbose=[0],
)

xgboost_params = dict(
    learning_rate=[0.01, 0.05, 0.1, 0.2],
    max_depth=[3, 5, 7, 10],
    subsample=[0.5, 0.8, 1.0],
    reg_alpha=[0, 0.1, 0.5, 1.0],     
    reg_lambda=[0, 0.1, 0.5, 1.0],      
    min_child_weight=[1, 3, 5],
    gamma=[0, 0.1, 0.3],
)

lgbm_params = dict(
    learning_rate=[0.01, 0.05, 0.1, 0.2],
    num_leaves=[20, 31, 40, 50],
    max_depth=[3, 5, 10, 15],
    min_child_samples=[10, 20, 50],
    subsample=[0.5, 0.8, 1.0],
    reg_alpha=[0, 0.1, 0.5, 1.0],     
    reg_lambda=[0, 0.1, 0.5, 1.0],     
    verbose=[-1],
)

mlp_params = dict(
    hidden_layer_sizes=[
        (32,),
        (64,),
        (32, 16),
        (64, 32),
        (32, 16, 8),
        (64, 32, 16),
    ],
    activation=["relu", "tanh"],
    solver=["adam", "sgd"],
    alpha=[ 0.001, 0.01, 0.1],           # L2 regularization
    learning_rate=["constant", "adaptive"],       # only used when solver="sgd"
    learning_rate_init=[0.001, 0.01, 0.05],
    max_iter=[1000, 1500, 2000],
    early_stopping=[True],
)

models = {
    "decision_tree":  (DecisionTreeRegressor(),  decision_tree_params),
    "random_forest":  (RandomForestRegressor(),  random_forest_params),
    "svr":            (SVR(), svr_params),
    "catboost":       (CatBoostRegressor(), catboost_params),
    "xgboost":        (XGBRegressor(), xgboost_params),
    "lgbm":           (LGBMRegressor(), lgbm_params),
    "mlp":            (MLPRegressor(), mlp_params)
}


for name, (model, params) in models.items():
    gs = GridSearchCV(model, param_grid=params, cv=2, n_jobs=12)
    gs.fit(data, targets)
    all_best_params[name] = gs.best_params_
    print(f"{name}: {gs.best_params_}")

with open("best_params_all.json", "w") as f:
    json.dump(all_best_params, f, indent=4)

