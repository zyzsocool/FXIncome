import pandas as pd
import numpy as np
import logging
import datetime
import os
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats
from fxincome.const import RFC_PARAM
from fxincome.ml import process_data
from fxincome.logger import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

"""
用最新的数据检验模型，
    Args:
        df(DataFrame): 检验数据，含日期，含labels
        model: RandomForest Classifier Model
    Returns:
        history_result(DataFrame): 'date', 'result', 'predict', 'actual', 'down_proba', 'up_proba'
"""


def val_model(model, df):
    X = df[RFC_PARAM.TRAIN_FEATS]
    y = df[RFC_PARAM.LABELS].squeeze().to_numpy()
    logger.info(model.get_params)
    test_pred = model.predict(X)
    model_score = model.score(X, y)
    logger.info("Test report: ")
    print(classification_report(y, test_pred))
    logger.info(f"Test score is: {model_score}")
    logger.info("Feature importances")
    for f_name, score in sorted(zip(RFC_PARAM.TRAIN_FEATS, model.feature_importances_)):
        logger.info(f"{f_name}, {round(score, 2)}")
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
                                          select_features=RFC_PARAM.ALL_FEATS,
                                          future_period=future_period,
                                          label_type=label_type,
                                          dropna=False)
    today = df.date.iloc[-1]
    last_x = df[RFC_PARAM.TRAIN_FEATS].tail(1)
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

    logger.info(f"预测下跌概率：{round(proba[0][0], 2)}，预测上涨概率：{round(proba[0][1], 2)}")
    return pred, proba


if __name__ == '__main__':
    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'

    sample_file = r'd:\ProjectRicequant\fxincome\fxincome_features_latest.csv'
    sample_df = pd.read_csv(sample_file, parse_dates=['date'])
    test_df = process_data.feature_engineering(sample_df,
                                               select_features=RFC_PARAM.ALL_FEATS,
                                               future_period=1,
                                               label_type='fwd')
    test_df.to_csv(os.path.join(ROOT_PATH, 'test_df.csv'), index=False, encoding='utf-8')
    model = joblib.load(f"models/0.701-1d_fwd-RandomSearch-20210606-2035.pkl")
    # model = joblib.load(f"models/0.613-1d_fwd-RandomSearch-20210605-2150.pkl")
    # model = joblib.load(f"models/0.714-5d_avg-RandomSearch-20210606-0004.pkl")
    history_result = val_model(model, test_df)
    history_result.to_csv(os.path.join(ROOT_PATH, 'history_result.csv'), index=False, encoding='utf-8')
    pred_future(model, sample_df, future_period=1, label_type='fwd')

