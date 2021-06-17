# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from fxincome.logger import logger
from fxincome.const import TBOND_PARAM


def label(row):
    """
    生成'taget'列的辅助函数，target即预测目标（label）
    """
    if pd.isnull(row.future) or pd.isnull(row.close):
        return np.nan
    elif row.future > row.close:
        return 1
    else:
        return 0


def feature_engineering(df, select_features, future_period, label_type='fwd', dropna=True):
    """
    处理10年国债收益率的features和labels，其中label只有1列，名字为'target'
    'target'为未来第future_period天的收盘值或未来对future_period天平均值比当日的涨跌情况，涨为1，跌为0

        Args:
            df(DataFrame): 待处理的原始数据dataframe，不含labels
            select_features(List): 字符串列表。只选择列表中的features。
            future_period(int): label的观察期，用于对比当日的收盘价，生成涨跌label。
            label_type(str): 生成label的规则，只限于'fwd'或'avg'，默认为'fwd'
                'fwd': label为未来第n天对比当日收盘价的涨跌
                'avg': label为未来n天平均值对比当日收盘价的涨跌
            dropna(Boolean): 是否去除带有空值的行，默认为去除。
        Returns:
            df(DataFrame)
    """
    df = df.rename(columns={'b19_c_ytm': 'close', 'b19_o_ytm': 'open', 'b19_h_ytm': 'high', 'b19_l_ytm': 'low',
                            'b19_amt': 'amount', 'b19_ttm': 'ttm'
                            })
    # 生成labels
    if label_type == 'fwd':
        df['future'] = df['close'].shift(-future_period)
    elif label_type == 'avg':
        df['future'] = df.close.rolling(pd.api.indexers.FixedForwardWindowIndexer(window_size=future_period)).mean()
    else:
        raise NotImplementedError("Unknown label_type")

    df['target'] = df.apply(lambda x: label(x), axis=1)

    # 生成features
    # 收盘收益ytm变种
    df['pct_chg'] = df.close.pct_change()
    df['avg_chg_5'] = (df.close - df.close.rolling(5).mean()) / df.close.rolling(5).mean()
    df['avg_chg_20'] = (df.close - df.close.rolling(20).mean()) / df.close.rolling(20).mean()
    df['volaty'] = (df.low - df.high) / df.close
    # 流动性指标变种
    df['fr007_chg_5'] = (df.fr007 - df.fr007.rolling(5).mean()) / df.fr007.rolling(5).mean()
    df['fr007_1y_chg_5'] = (df.fr007_1y - df.fr007_1y.rolling(5).mean()) / df.fr007_1y.rolling(5).mean()
    # 收盘ytm与其他各种指标之间的差值
    df['spread_t1y'] = df.close - df.t1y
    df['spread_t10y'] = df.close - df.t10y
    df['spread_fr007'] = df.close - df.fr007
    df['spread_fr007_1y'] = df.close - df.fr007_1y
    df['spread_fr007_5y'] = df.close - df.fr007_5y
    df['spread_usdcny'] = df.close - df.usdcny
    # 其他各种指标之间的差值
    df['spread_fr007_5y_fr007_1y'] = df.fr007_5y - df.fr007_1y
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
    df = feature_engineering(df, TBOND_PARAM.ALL_FEATS, future_period=1, label_type='fwd')
    df.to_csv(os.path.join(ROOT_PATH, DEST_NAME), index=False, encoding='utf-8')
