import pandas as pd
import numpy as np
import datetime
import os
import joblib
import matplotlib.pyplot as plt
import sklearn as sk
import xgboost
from fxincome.const import MTM_PARAM
from fxincome.ml import mtm_process_data, mtm_model
from fxincome.logger import logger
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier

"""
展示决策树模型中的任意一颗树，如果是XGBClassifier，还将pot_importance
    Args:
        model: 决策树模型，可以为RandomForestClassifier 或 XGBClassifier 
    Returns:
        none
"""

def show_tree(model):
    plt.rcParams.update({'figure.figsize': (20, 16)})
    plt.rcParams.update({'font.size': 12})
    if isinstance(model, RandomForestClassifier):
        tree = model.estimators_[0].tree_
        logger.info(f"Tree depth: {tree.max_depth}")
        sk.tree.plot_tree(tree, feature_names=MTM_PARAM.TRAIN_FEATS, filled=True)

    elif isinstance(model, xgboost.XGBClassifier):
        logger.info(f"Tree depth: {model.max_depth}")
        xgboost.plot_tree(model, num_trees=12)
        xgboost.plot_importance(model, importance_type='weight')
    else:
        raise NotImplementedError("Unknown Tree Model!")
    plt.show()

"""
用最新的数据检验模型。输入可为多个模型。对于每个模型，依次显示模型的name， params, test_report, score, feature_importances（如有）
只有树状模型才显示feature_importance，目前只支持'XGBClassifier'和'RandomForestClassifier' 
    Args:
        models(List): A list of models. 每个model必须具有以下methods: 
            predict(X), score(X,y), predict_proba(X)
        df(DataFrame): 检验数据，含日期，含labels，需要做好预处理
    Returns:
        history_result(DataFrame): 'date', 'actual', [每个model预测的'result', 'pred', 'actual', 'down_proba', 'up_proba']
"""


def val_models(models, df):
    X = df[MTM_PARAM.TRAIN_FEATS]
    y = df[MTM_PARAM.LABELS].squeeze().to_numpy()
    df = df[['date']]
    names = []
    col_names = ['date']
    for model in models:
        name = model.__class__.__name__
        if name == 'Pipeline':
            name = model.steps[-1][0]
        logger.info(f"Model: {name}")
        logger.info(model.get_params)
        test_pred = model.predict(X)
        model_score = model.score(X, y)
        print(classification_report(y, test_pred))
        logger.info(f"Test score is: {model_score}")
        if name in ['XGBClassifier', 'RandomForestClassifier']:
            logger.info("Feature importances")
            for f_name, score in sorted(zip(MTM_PARAM.TRAIN_FEATS, model.feature_importances_)):
                logger.info(f"{f_name}, {round(float(score), 2)}")
        probs = model.predict_proba(X)
        df.insert(len(df.columns), column=f'{name}_pred', value=test_pred)
        df.insert(len(df.columns), column=f'{name}_actual', value=y)
        df = df.copy()
        df[name] = df.apply(lambda x: 'Right' if x[f'{name}_pred'] == x[f'{name}_actual'] else 'Wrong', axis=1)
        df[f'{name}_down'], df[f'{name}_up'] = probs[:, 0], probs[:, 1]
        names.append(name)
    for name in names:
        col_names.append(name)
        col_names.append(f'{name}_pred')
        col_names.append(f'{name}_actual')
        col_names.append(f'{name}_down')
        col_names.append(f'{name}_up')
    history_result = df[col_names].copy()
    history_result.loc['average'] = history_result.mean(numeric_only=True)
    return history_result

"""
用最新的数据检验模型，
    Args:
        df(DataFrame): 检验数据，必须含labels，无需做预处理
        models(List): A list of models. 每个model必须具有以下methods: predict(X), predict_proba(X)
        future_period(int): label的观察期，用于对比当日的收盘价，生成涨跌label。
        label_type(str): 预测规则，只限于'fwd'或'avg'，默认为'fwd'
            'fwd': 预测未来第n天对比当日收盘价的涨跌
            'avg': 预测未来n天平均值对比当日收盘价的涨跌
    Returns:
        preds(List): 2D array-like of shape(n_models, n_dates), model顺序与输入的顺序一样；0为下跌，1为上涨
        probas(List): 3D array-like of shape(n_models, n_dates, 2), model顺序与输入的顺序一样；最后一项为[下跌概率， 上涨概率]
"""


def pred_future(models, df, future_period=1, label_type='fwd'):
    df = mtm_process_data.feature_engineering(df,
                                          select_features=MTM_PARAM.ALL_FEATS,
                                          future_period=future_period,
                                          label_type=label_type,
                                          dropna=False)
    today = df.date.iloc[-1].date()
    last_x = df[MTM_PARAM.TRAIN_FEATS].tail(1)
    preds = []
    probas = []
    for model in models:
        name = model.__class__.__name__
        if name == 'Pipeline':
            name = model.steps[-1][0]
        pred = model.predict(last_x)
        proba = model.predict_proba(last_x)
        if pred[0] == 0:
            if label_type == 'fwd':
                logger.info(f"{name} - {today} - 10年国债收益率在{future_period}个交易日后将下跌")
            elif label_type == 'avg':
                logger.info(f"{name} - {today} - 10年国债收益率在未来{future_period}个交易日的均值对比今天将下跌")
            else:
                raise NotImplementedError("Unknown label_type")
        elif pred[0] == 1:
            if label_type == 'fwd':
                logger.info(f"{name} - {today} - 10年国债收益率在{future_period}个交易日后将上涨")
            elif label_type == 'avg':
                logger.info(f"{name} - {today} - 10年国债收益率在未来{future_period}个交易日的均值对比今天将上涨")
            else:
                raise NotImplementedError("Unknown label_type")
        else:
            logger.info(f"Unknown result: {name} - {pred[0]}")

        logger.info(f"预测下跌概率：{round(float(proba[0][0]), 2)}，预测上涨概率：{round(float(proba[0][1]), 2)}")
        preds.append(pred)
        probas.append(proba)
    return preds, probas


if __name__ == '__main__':
    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'

    sample_file = r'd:\ProjectRicequant\fxincome\fxincome_features_latest.csv'
    sample_df = pd.read_csv(sample_file, parse_dates=['date'])
    test_df = mtm_process_data.feature_engineering(sample_df,
                                               select_features=MTM_PARAM.ALL_FEATS,
                                               future_period=1,
                                               label_type='fwd')
    test_df.to_csv(os.path.join(ROOT_PATH, 'test_df.csv'), index=False, encoding='utf-8')
    train_X, train_y, val_X, val_y, test_X, test_y = mtm_model.generate_dataset(test_df, root_path=ROOT_PATH,
                                                                                val_ratio=0.1, test_ratio=0.1)
    svm_model = joblib.load(f"models/0.59-1d_fwd-SVM-20210614-1153-v2018.pkl")
    rfc_model = joblib.load(f"models/0.615-1d_fwd-RFC-20210610-1744-v2018.pkl")
    xgb_model = joblib.load(f"models/0.656-1d_fwd-XGB-20210611-2231-v2016.pkl")
    pol_model = joblib.load(f"models/0.586-1d_fwd-POLY-20210616-0949-v2016.pkl")

    vote_model = EnsembleVoteClassifier(clfs=[xgb_model, rfc_model, svm_model],
                                   weights=[1, 1, 1], voting='hard', fit_base_estimators=False)
    vote_model.fit(val_X, val_y)
    history_result = val_models([vote_model, xgb_model, rfc_model, svm_model], test_df)
    pred_future([vote_model, xgb_model, rfc_model, pol_model], sample_df, future_period=1, label_type='fwd')
    history_result.to_csv(os.path.join(ROOT_PATH, 'history_result.csv'), index=False, encoding='utf-8')
