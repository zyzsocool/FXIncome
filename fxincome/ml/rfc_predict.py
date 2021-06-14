import pandas as pd
import numpy as np
import datetime
import os
import joblib
import matplotlib.pyplot as plt
import sklearn as sk
import xgboost
from fxincome.const import MTM_PARAM
from fxincome.ml import process_data, rfc_model
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
用最新的数据检验模型，
    Args:
        model: RandomForest Classifier Model
        df(DataFrame): 检验数据，含日期，含labels
        feat_importance(boolean): 是否显示feature_importance。只有树状模型才有feature_importance。默认为False
    Returns:
        history_result(DataFrame): 'date', 'result', 'predict', 'actual', 'down_proba', 'up_proba'
"""


def val_model(model, df, feat_importance=False):
    X = df[MTM_PARAM.TRAIN_FEATS]
    y = df[MTM_PARAM.LABELS].squeeze().to_numpy()
    logger.info(model.get_params)
    test_pred = model.predict(X)
    model_score = model.score(X, y)
    logger.info("Test report: ")
    print(classification_report(y, test_pred))
    logger.info(f"Test score is: {model_score}")
    if feat_importance:
        logger.info("Feature importances")
        for f_name, score in sorted(zip(MTM_PARAM.TRAIN_FEATS, model.feature_importances_)):
            logger.info(f"{f_name}, {round(float(score), 2)}")

    probs = model.predict_proba(X)
    df = df[['date']]
    df.insert(1, column='predict', value=test_pred)
    df.insert(2, column='actual', value=y)
    df = df.copy()
    df['result'] = df.apply(lambda x: 'Right' if x.predict == x.actual else 'Wrong', axis=1)
    df['down_proba'], df['up_proba'] = probs[:, 0], probs[:, 1]
    history_result = df[['date', 'result', 'predict', 'actual', 'down_proba', 'up_proba']]
    return history_result

"""
用最新的数据检验模型，
    Args:
        df(DataFrame): 检验数据，含labels
        model: RandomForest Classifier Model
        future_period(int): label的观察期，用于对比当日的收盘价，生成涨跌label。
        label_type(str): 预测规则，只限于'fwd'或'avg'，默认为'fwd'
            'fwd': 预测未来第n天对比当日收盘价的涨跌
            'avg': 预测未来n天平均值对比当日收盘价的涨跌
    Returns:
        test_pred(List): 0为下跌，1为上涨
        proba(List): a list of [下跌概率， 上涨概率]
"""


def pred_future(model, df, future_period=1, label_type='fwd'):
    df = process_data.feature_engineering(df,
                                          select_features=MTM_PARAM.ALL_FEATS,
                                          future_period=future_period,
                                          label_type=label_type,
                                          dropna=False)
    today = df.date.iloc[-1]
    last_x = df[MTM_PARAM.TRAIN_FEATS].tail(1)
    pred = model.predict(last_x)
    proba = model.predict_proba(last_x)
    if pred[0] == 0:
        if label_type == 'fwd':
            logger.info(f"{today} - 10年国债收益率在{future_period}个交易日后将下跌")
        elif label_type == 'avg':
            logger.info(f"{today} - 10年国债收益率在未来{future_period}个交易日的均值对比今天将下跌")
        else:
            raise NotImplementedError("Unknown label_type")
    elif pred[0] == 1:
        if label_type == 'fwd':
            logger.info(f"{today} - 10年国债收益率在{future_period}个交易日后将上涨")
        elif label_type == 'avg':
            logger.info(f"{today} - 10年国债收益率在未来{future_period}个交易日的均值对比今天将上涨")
        else:
            raise NotImplementedError("Unknown label_type")
    else:
        logger.info(f"Unknown result: {pred[0]}")

    logger.info(f"预测下跌概率：{round(float(proba[0][0]), 2)}，预测上涨概率：{round(float(proba[0][1]), 2)}")
    return pred, proba


if __name__ == '__main__':
    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'

    sample_file = r'd:\ProjectRicequant\fxincome\fxincome_features_latest.csv'
    sample_df = pd.read_csv(sample_file, parse_dates=['date'])
    test_df = process_data.feature_engineering(sample_df,
                                               select_features=MTM_PARAM.ALL_FEATS,
                                               future_period=1,
                                               label_type='fwd')
    test_df.to_csv(os.path.join(ROOT_PATH, 'test_df.csv'), index=False, encoding='utf-8')
    train_X, train_y, val_X, val_y, test_X, test_y = rfc_model.generate_dataset(test_df, root_path=ROOT_PATH,
                                                                                val_ratio=0.1, test_ratio=0.1)
    # lr_model = joblib.load(f"models/0.586-1d_fwd-LR-20210614-1813-v2016.pkl")
    svm_model = joblib.load(f"models/0.59-1d_fwd-SVM-20210614-1153-v2018.pkl")
    rfc_model = joblib.load(f"models/0.615-1d_fwd-RFC-20210610-1744-v2018.pkl")
    xgb_model = joblib.load(f"models/0.656-1d_fwd-XGB-20210611-2231-v2016.pkl")

    model = EnsembleVoteClassifier(clfs=[svm_model, rfc_model, xgb_model],
                                   weights=[1, 1, 1], voting='soft', fit_base_estimators=False)
    model.fit(val_X, val_y)
    history_result = val_model(model, test_df)
    pred_future(model, sample_df, future_period=1, label_type='fwd')
    history_result.to_csv(os.path.join(ROOT_PATH, 'history_result.csv'), index=False, encoding='utf-8')
