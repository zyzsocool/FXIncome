# coding: utf-8

import joblib
import numpy as np
import pandas as pd
import os
from fxincome import logger
from fxincome.const import SPREAD
from fxincome.spread import preprocess_data
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix

# train and validation set
days_back = 5
n_samples = 190
leg1_code = "180210"
leg2_code = "190205"
df_target = preprocess_data.feature_engineering(leg1_code, leg2_code, days_back, n_samples)
for i in range(SPREAD.CDB_CODES.index(leg2_code), 10):
    df_target = pd.concat([df_target,
                           preprocess_data.feature_engineering(SPREAD.CDB_CODES[i], SPREAD.CDB_CODES[i + 1], days_back,
                                                               n_samples)], axis=0)
days_forward = 20
Y = preprocess_data.target_20D_0bp(leg1_code, leg2_code, days_back, n_samples, days_forward)
for i in range(SPREAD.CDB_CODES.index(leg2_code), 10):
    Y = pd.concat([Y, preprocess_data.target_20D_0bp(SPREAD.CDB_CODES[i], SPREAD.CDB_CODES[i + 1], days_back, n_samples,
                                                     days_forward)], axis=0)

# test set
n_samples = 135
leg1_code = '220205'
leg2_code = '220210'
df_test = preprocess_data.feature_engineering(leg1_code, leg2_code, days_back, n_samples)
leg1_code = '220210'
leg2_code = '220215'
df_test = pd.concat([df_test, preprocess_data.feature_engineering(leg1_code, leg2_code, days_back, n_samples)])
leg1_code = '220205'
leg2_code = '220210'
Y_test = preprocess_data.target_20D_0bp(leg1_code, leg2_code, days_back, n_samples, days_forward)
leg1_code = '220210'
leg2_code = '220215'
Y_test = pd.concat([Y_test, preprocess_data.target_20D_0bp(leg1_code, leg2_code, days_back, n_samples, days_forward)])

random_seed = 10

X_train, X_test, y_train, y_test = train_test_split(df_target.drop(columns=['DATE'])
                                                    , Y.drop(columns=['DATE'])
                                                    , random_state=random_seed
                                                    , test_size=0.2)

# GridSampler：网格调参
# RandomSampler：随机调参
# TPESampler：贝叶斯调参
# seed：搜索超参数的随机数种子(固定一个整数即可)
sampler = TPESampler(seed=random_seed)


# 定义Objective
def objective(trial):
    # 装载数据
    dtrain = xgb.DMatrix(X_train, label=y_train)
    X_test_DMatrix = xgb.DMatrix(X_test)

    # 定义参数搜索空间
    # 选择型搜索方式 【 # 从MomentumSGD和Adam二者中选】
    # trail.suggest_categorical('optimizer',['MomentumSGD','Adam'])

    # 整型搜索方式 【 # 从1～3范围内的int选择】
    # trail.suggest_int('max_depth',1,3)

    # 连续均匀采样搜索方式 【 # 从0～1.0之间的浮点数进行均匀采样】
    # trail.suggest_uniform('dropout_rate',0.0,1.0)

    # 对数均匀采样方式【 # 从log(1e-5)~log(1e-2)均匀分布中采样结果再取e的自然指数】
    # trail.suggest_loguniform('learning_rate,1e-5,1e-2')

    # 离散均匀采样方式【# 以0.1为步长拆分0～1后的离散均匀分布中采样】
    # trail.suggest_discrete_uniform('drop_path_rate',0.0,1.0,0.1)

    param = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        # 由于乳腺癌数据集不是极度不平衡数据集，所以不需要指定scale_pos_weight参数
        # "scale_pos_weight":None,
        'n_estimators': trial.suggest_int('n_estimators', 10, 300),
        "max_depth": trial.suggest_int('max_depth', 3, 8),
        "min_child_weight": trial.suggest_uniform('min_child_weight', 0.1, 3.0),
        'gamma': trial.suggest_uniform('gamma', 0.0, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'alpha': trial.suggest_uniform('alpha', 0.0, 10.0),
        'lambda': trial.suggest_uniform('lambda', 0.0, 10.0),
        'eta': trial.suggest_uniform('eta', 0.0, 3.0),
    }

    # 训练模型
    xgb_model = xgb.train(param, dtrain)

    # 预测
    y_pred = xgb_model.predict(X_test_DMatrix)
    # 区分0，1
    # y_pred01 = np.where(y_pred > 0.5,1,0)

    # 返回auc作为度量指标
    return roc_auc_score(y_test, y_pred)


# 调参try
# direction可选参数：maximize、minimize
# maximize按照最大值，例如准确率这种是越大越好
# minimize按照最小值，例如logloss这种是越小越好
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100)

# 输出模型的最好结参数
print(study.best_params)

# 输出模型最优参数下的auc
print(study.best_value)

best_bound = 0
best_value1 = 0
best_value2 = 0
for i in range(0, 100):
    def xgb_clf(train_x, train_y, test_x, test_y, i):

        # 装载数据
        train_DMatrix = xgb.DMatrix(train_x, label=train_y)
        test_x_DMatrix = xgb.DMatrix(test_x)

        # 初始参数
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': study.best_params['n_estimators'],
            'max_depth': study.best_params['max_depth'],
            'min_child_weight': study.best_params['min_child_weight'],
            'gamma': study.best_params['gamma'],
            'subsample': study.best_params['subsample'],
            'colsample_bytree': study.best_params['colsample_bytree'],
            'alpha': study.best_params['alpha'],
            'lambda': study.best_params['lambda'],
            'eta': study.best_params['eta'],
            'silent': True,

        }

        # 训练模型
        xgb_model = xgb.train(params, train_DMatrix)

        # 预测测试集数据
        y_test_pred = xgb_model.predict(test_x_DMatrix)

        return np.where(y_test_pred > 0.01 * i, 1, 0), roc_auc_score(test_y, y_test_pred)


    y_test_pred, y_test_pred_auc = xgb_clf(X_train, y_train, X_test, y_test, i)
    if (recall_score(y_test, y_test_pred.reshape(len(y_test_pred), 1)) + accuracy_score(y_test, y_test_pred.reshape(
            len(y_test_pred), 1)) >= (best_value1 + best_value2)):
        best_bound = 0.01 * i
        best_value1 = accuracy_score(y_test, y_test_pred.reshape(len(y_test_pred), 1))
        best_value2 = recall_score(y_test, y_test_pred.reshape(len(y_test_pred), 1))
print(best_value1, best_value2, best_bound, y_test_pred_auc)


def xgb_clf(train_x, train_y, test_x, test_y, best_bound):
    """
    xgb分类模型
    param:
        train_x:输入训练集特征
        train_y:输入训练集标签
        test_x：测试集特征
    return:
        测试集上的性能
        测试集上的预测值
    """

    # 装载数据
    train_DMatrix = xgb.DMatrix(train_x, label=train_y)  # ,feature_names=columns_names)
    test_x_DMatrix = xgb.DMatrix(test_x)  # ,feature_names=columns_names)

    # 初始参数
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': study.best_params['n_estimators'],
        'max_depth': study.best_params['max_depth'],
        'min_child_weight': study.best_params['min_child_weight'],
        'gamma': study.best_params['gamma'],
        'subsample': study.best_params['subsample'],
        'colsample_bytree': study.best_params['colsample_bytree'],
        'alpha': study.best_params['alpha'],
        'lambda': study.best_params['lambda'],
        'eta': study.best_params['eta'],
        'silent': True,

    }

    # 训练模型
    xgb_model = xgb.train(params, train_DMatrix)

    # 预测测试集数据
    y_test_pred = xgb_model.predict(test_x_DMatrix)
    y_test_pred_train = xgb_model.predict(train_DMatrix)

    return np.where(y_test_pred > best_bound, 1, 0), roc_auc_score(test_y, y_test_pred), np.where(
        y_test_pred_train > best_bound, 1, 0)  # xgb_model.get_score(importance_type="gain")


y_test_pred, y_test_pred_auc, y_test_pred_train = xgb_clf(X_train, y_train, X_test, y_test, best_bound)
print(y_test_pred_auc, accuracy_score(y_test, y_test_pred.reshape(len(y_test_pred), 1)),
      accuracy_score(y_train, y_test_pred_train.reshape(len(y_test_pred_train), 1)))

y_test_pred, y_test_pred_auc, feature_importance = xgb_clf(X_train, y_train, df_test.drop(columns=['DATE']),
                                                           Y_test.drop(columns=['DATE']), best_bound)
y_test_pred_auc

recall_score(Y_test.drop(columns=['DATE']), y_test_pred.reshape(len(y_test_pred), 1))

accuracy_score(Y_test.drop(columns=['DATE']), y_test_pred.reshape(len(y_test_pred), 1))

train_DMatrix = xgb.DMatrix(X_train, label=y_train)
test_x_DMatrix = xgb.DMatrix(X_test)
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': study.best_params['n_estimators'],
    'max_depth': study.best_params['max_depth'],
    'min_child_weight': study.best_params['min_child_weight'],
    'gamma': study.best_params['gamma'],
    'subsample': study.best_params['subsample'],
    'colsample_bytree': study.best_params['colsample_bytree'],
    'alpha': study.best_params['alpha'],
    'lambda': study.best_params['lambda'],
    'eta': study.best_params['eta'],
    'silent': True, }
xgb_model = xgb.train(params, train_DMatrix)
y_test_pred = xgb_model.predict(test_x_DMatrix)
roc_auc_score(y_test, y_test_pred)

dirs = r'C:\Users\DELL\Desktop\huangqiwen'
if not os.path.exists(dirs):
    os.makedirs(dirs)

joblib.dump(xgb_model, dirs + '/xgb_model_30D_0bp.pkl')
