# -*- coding: utf-8 -*-
import os
import random
import datetime
from collections import deque
import numpy as np
import pandas as pd
import fxincome.ml.tbond_process_data
from fxincome import logger
from fxincome.const import TBOND_PARAM

"""
    为训练神经网络做数据预处理
"""

def feature_engineering(df, select_features, future_period, label_type='fwd', dropna=True):
    """
    处理国债收益率的features和labels，其中label只有1列，名字为'target'
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

    df['target'] = df.apply(lambda x: fxincome.ml.tbond_process_data.label(x), axis=1)

    # 生成features
    # 收盘收益ytm变种
    df['pct_chg'] = df.close.pct_change()
    df['avg_chg_5'] = (df.close - df.close.rolling(5).mean()) / df.close.rolling(5).mean()
    df['avg_chg_10'] = (df.close - df.close.rolling(10).mean()) / df.close.rolling(10).mean()
    df['avg_chg_20'] = (df.close - df.close.rolling(20).mean()) / df.close.rolling(20).mean()
    df['volaty'] = (df.low - df.high) / df.close
    # 流动性指标变种
    df['fr007_chg_5'] = (df.fr007 - df.fr007.rolling(5).mean()) / df.fr007.rolling(5).mean()
    df['fr007_1y_chg_5'] = (df.fr007_1y - df.fr007_1y.rolling(5).mean()) / df.fr007_1y.rolling(5).mean()
    # 其他指标变种
    df['t10y_chg_5'] = (df.t10y - df.t10y.rolling(5).mean()) / df.t10y.rolling(5).mean()
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


def pre_process(df, scaled_features, percentile=0.10, scale_type='zscore'):
    """
    原始features做特征工程之后，为训练模型做准备。
    样本分成训练集、验证集和测试集，然后做scaling。
    划分数据集方法：按日期顺序划分，从早到晚依次为test,val，test。
                 设percentile为x，则 test = 1 - 2*x, val = x, test = x

    Scaling：针对scaled_features做scale，在划分数据集之后再scale
        Args:
            df(DataFrame): 待处理的dataframe，含labels
            scaled_features(List): 字符串列表。需要做scaling的features。
            percentile(float):  划分比例，test = 1 - 2*x, val = x, test = x
            scale_type('minmax', 'zscore'): 采用什么归一化方法，可选min-max和 z score，默认为z score
        Returns:
            train_df(DataFrame)
            val_df(DataFrame)
            test_df(DataFrame)
            train_stats(Dict): 训练集做Scaling的统计特征，数据结构为Dict of Dict
                            对于zscore， {feature1: {mean: float, std: float}, feature2: ...}
                            对于minmax， {feature1: {min: float, max: float}, feature2: ...}
    """
    logger.info(f"Before pre_data() splitting, sample size is {len(df)}")
    # SCALED_FEATURES不能为空
    df = df.dropna(subset=TBOND_PARAM.SCALED_FEATS)
    df = df.sort_values('date', ascending=True)
    dates = sorted(df.date.values)

    test_date = dates[-int(percentile * len(dates))]
    val_date = dates[-int(2 * percentile * len(dates))]

    test_df = df.query('date >= @test_date').copy()
    val_df = df.query('date >= @val_date & date < @test_date').copy()
    train_df = df.query('date < @val_date').copy()
    logger.info(f"After pre_data() splitting, sample size is {len(train_df)+len(test_df)+len(val_df)}")
    logger.info(f"train size is {len(train_df)} val size is {len(val_df)}  test size is {len(test_df)}")

    train_df, train_stats = scale(train_df, scaled_features, type=scale_type)
    val_df, stats = scale(val_df, scaled_features, stats=train_stats, type=scale_type)
    test_df, stats = scale(test_df, scaled_features, stats=train_stats, type=scale_type)

    return train_df, val_df, test_df, train_stats


def scale(df, features, stats=None, type='zscore'):
    """
    针对scaled_features做scale
        Args:
            df(DataFrame): 待处理的dataframe，只对里面的features做scale
            features(List): 字符串列表。需要做scaling的features。
            stats(Dict):    Scaling的统计特征，数据结构为Dict of Dict
                            对于zscore， {feature1: {mean: float, std: float}, feature2: ...}
                            对于minmax， {feature1: {min: float, max: float}, feature2: ...}
                            默认为None，即未训练，需要由本mehtod生成统计特征。
                            若不为None，即已训练，需要用stats里的统计特征。
            type('minmax', 'zscore'): 采用什么归一化方法，可选min-max和 z score，默认为z score
        Returns:
            df(DataFrame)
            stats(Dict): 训练集做Scaling的统计特征，数据结构为Dict of Dict
                         对于zscore， {feature1: {mean: float, std: float}, feature2: ...}
                         对于minmax， {feature1: {min: float, max: float}, feature2: ...}
    """

    if stats:   # 已经训练好模型，必须用模型的统计要素
        trained = True
        if stats['type'] != type:
            raise ValueError(f"stats type({stats['type']}) != input scale type(type)")
    else:  # 未训练模型，需要自行生成train set的统计要素
        trained = False
        stats = {'type': type}
    for feature in features:
        # Default scaling method.
        # Use the std and mean of the train set to scale the val set and test set.
        if type == 'zscore':
            if trained:  # 已经训练好模型，必须用模型的统计要素
                std = stats[feature]['std']
                mean = stats[feature]['mean']
            else:  # 未训练模型，需要自行生成train set的统计要素
                std = df[feature].std()
                mean = df[feature].mean()
                stats[feature] = {'mean': mean, 'std': std}
            df[feature] = (df[feature] - mean) / std

        # Scaled to range: [0,1], but the scaled val set and test set may breach the range.
        # Use the max and min of the train set to scale the val set and test set.
        elif type == 'minmax':
            if trained: # 已经训练好模型，必须用模型的统计要素
                min = stats[feature]['min']
                max = stats[feature]['max']
            else: # 未训练模型，需要自行生成train set的统计要素
                min = df[feature].min()
                max = df[feature].max()
            df[feature] = df[feature] - min / (max - min)
        else:
            raise NotImplementedError('No such scale method!')
    return df, stats


def gen_trainset(df, columns: list, feature_outliners: list, seq_len=10, balance=True):
    """
    生成用于训练的X和Y
    输入：已完成特征工程、Scaling的dataframe，包含features和labels
    输出：符合RNN要求的X和Y。
    具备特征值异常检测，特征值超过95%阈值的样本将被舍弃。如不需要特征异常检测，可设置feature_outliners为空表。
        Args:
            df(DataFrame): 待处理的dataframe，含labels
            columns(List): 字符串列表，待处理的features和labels。
            feature_outliners(List): 字符串列表。列表中的特征将做异常检测，超过95%的样本视为异常。
            如果某日任意一个特征太异常，则舍弃整个序列，从下一日开始重新构造序列。
            seq_len(int): time steps的长度
            balance(Boolean): 平衡样本涨跌数量。随机抽取样本，并使得涨的样本 = 跌的样本。默认为True
        Returns:
            X(ndarray): 转换后，适合输入到RNN网络的numpy ndarray X[size, time steps, feature dimensions]
            y(ndarray): 转换后，适合输入到RNN网络的numpy ndarray y[size, time steps, label dimensions(1)]
    """

    df = df.dropna()  # cleanup again
    if len(df) < seq_len:
        raise ValueError(f'Samples ({len(df)}) less than seq_len ({seq_len})')
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=seq_len)  # These will be our actual sequences.
    quantiles = [df[f].quantile(0.95) for f in feature_outliners]  # 如果超过95%的阈值，则视为异常
    feature_quantiles = dict(zip(feature_outliners, quantiles))  # 异常值字典，key为feature, value为异常阈值

    df = df.sort_values(by='date', ascending=True)
    df = df[columns]
    prev_days.clear()
    for row in df.itertuples(index=False):  # iterate over the rows，即每日的数据，row是namedtuple
        if any([getattr(row, k) > v for k, v in feature_quantiles.items()]):
            # 如果某日任意一个特征太异常，则舍弃整个序列，从下一日开始重新构造序列。
            prev_days.clear()
            continue
        prev_days.append([n for n in row[:-1]])  # 以一个list的形式存储当日的所有列，除了'target'列
        if len(prev_days) == seq_len:  # make sure we have 预先设定数量SEQ_LEN的 sequences!
            # Features：长达SEQ_LEN的每日数据; Labels: 'target'
            sequential_data.append([np.array(prev_days), row.target])

    random.shuffle(sequential_data)
    # --- Balancing the data
    if balance:
        buys = []
        sells = []

        for seq, target in sequential_data:
            if target == 0:
                sells.append([seq, target])
            elif target == 1:
                buys.append([seq, target])

        random.shuffle(buys)
        random.shuffle(sells)

        lower = min(len(buys), len(sells))  # what's the shorter length?

        buys = buys[:lower]  # make sure both lists are only up to the shortest length.
        sells = sells[:lower]  # make sure both lists are only up to the shortest length.

        sequential_data = buys + sells
        random.shuffle(sequential_data)
    # --- End of balancing the data

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell)

    return np.array(X), np.array(y)  # make X and y numpy arrays!

def gen_pred_x(df, today, columns: list, seq_len=10):
    """
    生成用于预测的X
    输入：1. 已完成特征工程以及Scaling的dataframe，包含datetime和features
         2. 预测基准日
    输出：符合RNN要求的X。
        Args:
            df(DataFrame): 待处理的dataframe，包含datetime和features。可以含未来样本。
            today(datetime): 预测基准日，筛选出df里在此基准日（含）之前的输入样本。
            columns(List): 字符串列表，待处理的features。
            seq_len(int): time steps的长度
        Returns:
            X(ndarray): 2D numpy ndarray [time steps, feature dimensions]
    """

    df = df.dropna()  # cleanup again
    df = df[df.date <= today]
    if len(df) < seq_len:
        raise ValueError(f'Samples ({len(df)}) less than seq_len ({seq_len})')
    df = df.sort_values(by='date', ascending=False)  # 日期倒序排列，确保第1个row是today
    prev_days = deque(maxlen=seq_len)
    df = df[columns]
    for row in df.itertuples(index=False):  # iterate over the rows，即每日的数据，row是namedtuple
        prev_days.appendleft([n for n in row])  # 以一个list的形式存储当日的所有列, row进入prev_days后变成日期升序排列
        if len(prev_days) == seq_len:  # 回溯seq_len个日期
            break

    return np.array(prev_days)


def main():
    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'
    SRC_NAME = 'fxincome_features.csv'
    DEST_NAME = 'fxincome_processed.csv'

    df = pd.read_csv(os.path.join(ROOT_PATH, SRC_NAME), parse_dates=['date'])
    df = feature_engineering(df, TBOND_PARAM.ALL_FEATS, future_period=1, label_type='fwd')
    df.to_csv(os.path.join(ROOT_PATH, DEST_NAME), index=False, encoding='utf-8')

    train_df, val_df, test_df, stats = pre_process(df, TBOND_PARAM.SCALED_FEATS, scale_type='zscore')
    train_df.to_csv(os.path.join(ROOT_PATH, 'train_samples.csv'), index=False, encoding='utf-8')
    val_df.to_csv(os.path.join(ROOT_PATH, 'validation_samples.csv'), index=False, encoding='utf-8')
    test_df.to_csv(os.path.join(ROOT_PATH, 'test_samples.csv'), index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
