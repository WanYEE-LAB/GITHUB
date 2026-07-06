# 全局底层库线程限制
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 复用基线模块公共组件
from 基线ASR单预测 import (
    _check_input_valid,
    ASRPredictionNet
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ===================== 1. LightGBM 贝叶斯调参（空值特征专用） =====================
def bayes_opt_lightgbm_asr(
    X_train, y_train, X_test, y_test,
    n_trials=60,
    cv_folds=5,
    random_state=42
):
    _, y_train_arr = _check_input_valid(X_train, y_train, allow_nan_X=True)
    _, y_test_arr = _check_input_valid(X_test, y_test, allow_nan_X=True)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 0.1, log=True),
            "random_state": random_state,
            "n_jobs": 1,
            "verbose": -1
        }
        # 训练集内交叉验证，不碰测试集
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train_arr[train_idx], y_train_arr[val_idx]
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            cv_scores.append(r2_score(y_val, model.predict(X_val)))
        return np.mean(cv_scores)

    # 贝叶斯寻优
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # 最优参数全量重训
    best_params = study.best_params
    best_model = LGBMRegressor(**best_params, random_state=random_state, n_jobs=1, verbose=-1)
    best_model.fit(X_train, y_train_arr)

    # 测试集评估（log空间）
    y_pred_test = best_model.predict(X_test)
    r2_log = r2_score(y_test_arr, y_pred_test)
    mae_log = mean_absolute_error(y_test_arr, y_pred_test)

    return {
        "model_name": "LightGBM_ASR_贝叶斯调参",
        "best_params": best_params,
        "model": best_model,
        "r2_log": r2_log,
        "mae_log": mae_log,
        "best_cv_r2": study.best_value
    }

