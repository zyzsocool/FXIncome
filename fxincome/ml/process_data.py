# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from fxincome.logger import logger
from fxincome.const import MTM_PARAM

"""
生成'taget'列的辅助函数，target即预测目标（label）
"""
def label(row):
    if pd.isnull(row.future) or pd.isnull(row.close):
        return np.nan
    elif row.future > row.close:
        return 1
    else:
        return 0


def combine_fx_yields(path, filename='fixed_income.csv'):
    yields = pd.read_csv(os.path.join(path, 'Yields.csv'), parse_dates=['date'])
    usdcny = pd.read_csv(os.path.join(path, 'usdcny.csv'), parse_dates=['date'])
    df = pd.merge(yields, usdcny, on='date', how='left')
    df.usdcny = df.sort_values('date', ascending=True).usdcny.fillna(method='ffill')
    df.to_csv(os.path.join(path, filename), index=False, encoding='utf-8')

"""
处理10年国债收益率的features和labels，其中label只有1列，名字为'target'
'target'为未来第future_period天的收盘值或未来对future_period天平均值比当日的涨跌情况，涨为1，跌为0

    Args:
        df(DataFrame): 待处理的dataframe，不含labels
        select_features(List): 字符串列表。只选择列表中的features。
        future_period(int): label的观察期，用于对比当日的收盘价，生成涨跌label。
        label_type(str): 生成label的规则，只限于'fwd'或'avg'，默认为'fwd'
            'fwd': label为未来第n天对比当日收盘价的涨跌
            'avg': label为未来n天平均值对比当日收盘价的涨跌
        dropna(Boolean): 是否去除带有空值的行，默认为去除。
    Returns:
        df(DataFrame)
"""

def feature_engineering(df, select_features, future_period, label_type='fwd', dropna=True):
    df = df.rename(columns={'t10y': 'close'})
    # 生成labels
    if label_type == 'fwd':
        df['future'] = df['close'].shift(-future_period)
    elif label_type == 'avg':
        df['future'] = df.close.rolling(pd.api.indexers.FixedForwardWindowIndexer(window_size=future_period)).mean()
    else:
        raise NotImplementedError("Unknown label_type")

    df['target'] = df.apply(lambda x: label(x), axis=1)

    # 生成features
    # 10年国债收益率变种
    df['pct_chg'] = df.close.pct_change()
    df['avg_chg_5'] = (df.close - df.close.rolling(5).mean()) / df.close.rolling(5).mean()
    df['avg_chg_10'] = (df.close - df.close.rolling(10).mean()) / df.close.rolling(10).mean()
    df['avg_chg_20'] = (df.close - df.close.rolling(20).mean()) / df.close.rolling(20).mean()
    df['close_avg_5'] = df.close / df.close.rolling(5).mean()
    # 流动性指标变种
    df['fr007_chg_5'] = (df.fr007 - df.fr007.rolling(5).mean()) / df.fr007.rolling(5).mean()
    df['fr0071y_chg_5'] = (df.fr0071y - df.fr0071y.rolling(5).mean()) / df.fr0071y.rolling(5).mean()
    # 10年国债收益率与其他各种指标之间的差值
    df['spread_t1y'] = df.close - df.t1y
    df['spread_cdb10y'] = df.close - df.cdb10y
    df['spread_fr007'] = df.close - df.fr007
    df['spread_fr0071y'] = df.close - df.fr0071y
    df['spread_fr0075y'] = df.close - df.fr0075y
    df['spread_usdcny'] = df.close - df.usdcny
    # 其他各种指标之间的差值
    df['spread_fr0075y_fr0071y'] = df.fr0075y - df.fr0071y
    # 汇率变种
    df['usdcny_chg_5'] = (df.usdcny - df.usdcny.rolling(5).mean()) / df.usdcny.rolling(5).mean()
    df = df[select_features]
    logger.info(f"Before feature engineering, sample size is {len(df)}")
    if dropna:
        df = df.dropna()
    logger.info(f"After feature engineering, sample size is {len(df)}")
    return df

if __name__ == '__main__':

    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'
    SRC_NAME = 'fxincome_features.csv'
    DEST_NAME = 'fxincome_processed.csv'

    df = pd.read_csv(os.path.join(ROOT_PATH, SRC_NAME), parse_dates=['date'])
    df = feature_engineering(df, MTM_PARAM.ALL_FEATS, future_period=1, label_type='fwd')
    df.to_csv(os.path.join(ROOT_PATH, DEST_NAME), index=False, encoding='utf-8')