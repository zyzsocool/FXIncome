import pandas as pd
import numpy as np
import logging
import datetime
import os
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from fxincome.const import RFC_PARAM
from fxincome.logger import logger

def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input  # Contains all columns.
    y = df_input[[stratify_colname]]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp = train_test_split(X,
                                         stratify=y,
                                         test_size=(1.0 - frac_train),
                                         random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    # Do NOT stratify the validation and test dataframes
    df_val, df_test = train_test_split(df_temp, test_size=relative_frac_test, random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

"""
将所有样本分成训练集、验证集和测试集，同时写入csv
train, val, test 数据集的比例分别是：train = 1 - val_ratio - test_ratio, 
                                 val = val_ratio
                                 test = test_ratio
train set的labels分布比例遵从所有样本的labels的分布比例

    Args:
        main_df(DataFrame): 包含所有样本的dataframe，含labels
        val_ratio(float): 验证集的比例
        test_ratio(float): 测试集的比例 
    Returns:
        train_X(DataFrame)：column为FEATURES的DataFrame
        train_y(ndarray)：1D ndarray, 0 or 1
        val_X(DataFrame)：column为FEATURES的DataFrame
        val_y(ndarray)：1D ndarray, 0 or 1
        test_X(DataFrame)：column为FEATURES的DataFrame
        test_y(ndarray)：1D ndarray, 0 or 1
"""


def generate_dataset(main_df, val_ratio=0.1, test_ratio=0.1):
    train_df, test_df, val_df = split_stratified_into_train_val_test(main_df,
                                                                     stratify_colname='target',
                                                                     frac_train=1 - val_ratio - test_ratio,
                                                                     frac_val=val_ratio,
                                                                     frac_test=test_ratio)
    train_df.to_csv(os.path.join(ROOT_PATH, 'train_samples.csv'), index=False, encoding='utf-8')
    val_df.to_csv(os.path.join(ROOT_PATH, 'validation_samples.csv'), index=False, encoding='utf-8')
    test_df.to_csv(os.path.join(ROOT_PATH, 'test_samples.csv'), index=False, encoding='utf-8')

    train_X = train_df[RFC_PARAM.TRAIN_FEATS]
    train_y = train_df[RFC_PARAM.LABELS].squeeze().to_numpy()
    val_X = val_df[RFC_PARAM.TRAIN_FEATS]
    val_y = val_df[RFC_PARAM.LABELS].squeeze().to_numpy()
    test_X = test_df[RFC_PARAM.TRAIN_FEATS]
    test_y = test_df[RFC_PARAM.LABELS].squeeze().to_numpy()
    return train_X, train_y, val_X, val_y, test_X, test_y


def plot_graph(x_train, y_train, x_test, y_test, model):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(y_train, label='real')
    plt.plot(model.predict(x_train), label='predict')
    plt.legend(loc='upper right')
    plt.title('train')

    plt.subplot(3, 1, 2)
    plt.plot(y_test, label='real')
    plt.plot(model.predict(x_test), label='predict')
    plt.legend(loc='upper right')
    plt.title('test')

    plt.show()

"""
训练模型并输出报告，返回最佳模型, 及附带分数的模型名字

    Returns:
        model(RandomForestClassifer)：最佳的模型
        name_score(str): 最佳模型的来源 + 对应的分数。来源包括RandomizedSearchCV 和 GridSearchCV, 分数保留3位小数
"""

def train(train_X, train_y, val_X, val_y, test_X, test_y):
    combined_train_X = train_X.append(val_X)
    combined_train_y = np.append(train_y, val_y)
    logger.info(f"Size of train set X: {len(combined_train_X)}")
    clf = RandomForestClassifier(oob_score=True, n_jobs=-1)
    param_dist = {'n_estimators': stats.randint(50, 500),
                  'max_samples': stats.uniform(0.3, 0.7),
                  'min_samples_leaf': stats.loguniform(1e-4, 1e-2),
                  'max_leaf_nodes': stats.randint(20, 100),
                  'max_depth': stats.randint(10, 100),
                  }
    param_grid = {'n_estimators': np.arange(50, 500, 150),
                  'max_samples': np.linspace(0.3, 0.99, 4),
                  'min_samples_leaf': np.power(10, np.linspace(-4, -0.35, 4)),
                  'max_leaf_nodes': np.arange(10, 100, 40),
                  'max_depth': np.arange(10, 100, 40),
                  }
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=100,
                                       n_jobs=-1, pre_dispatch=64, verbose=4)
    random_search.fit(combined_train_X, combined_train_y)
    grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, pre_dispatch=64, verbose=4)
    grid_search.fit(combined_train_X, combined_train_y)
    random_model = random_search.best_estimator_
    grid_model = grid_search.best_estimator_
    logger.info("RandomSearch Report:\n _____________")
    random_score = report_model(random_model, test_X, test_y, train_X, train_y, val_X, val_y)
    logger.info("GridSearch Report:\n _____________")
    grid_score = report_model(grid_model, test_X, test_y, train_X, train_y, val_X, val_y)

    # rnd_clf = RandomForestClassifier(n_estimators=300,
    #                                  max_samples=0.7,
    #                                  min_samples_leaf=0.001,
    #                                  max_leaf_nodes=100,
    #                                  max_depth=20,
    #                                  oob_score=True,
    #                                  n_jobs=-1)
    if random_score > grid_score:
        return random_model, f"{random_score}-RandomSearch-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    return grid_model, f"{grid_score}-GridSearch-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

def random_train(train_X, train_y, val_X, val_y, test_X, test_y):
    combined_train_X = train_X.append(val_X)
    combined_train_y = np.append(train_y, val_y)
    logger.info(f"Size of train set X: {len(combined_train_X)}")
    clf = RandomForestClassifier(oob_score=True, n_jobs=-1)
    param_dist = {'n_estimators': stats.randint(50, 500),
                  'max_samples': stats.uniform(0.3, 0.7),
                  'min_samples_leaf': stats.loguniform(1e-3, 1e-1),
                  'max_leaf_nodes': stats.randint(20, 100),
                  'max_depth': stats.randint(10, 100),
                  }
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=200,
                                       n_jobs=-1, pre_dispatch=64, verbose=4)
    random_search.fit(combined_train_X, combined_train_y)
    random_model = random_search.best_estimator_
    logger.info("RandomSearch Report:\n _____________")
    random_score = report_model(random_model, test_X, test_y, train_X, train_y, val_X, val_y)
    return random_model, f"{random_score}-{PERIOD}-RandomSearch-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

def report_model(model, test_X, test_y, train_X, train_y, val_X, val_y):

    logger.info(model.get_params)
    test_pred = model.predict(test_X)
    model_score = model.score(test_X, test_y)
    logger.info("Test report: ")
    print(classification_report(test_y, test_pred))
    logger.info(f"Train oob_score is: {model.oob_score_}")
    logger.info(f"Train score is: {model.score(train_X, train_y)}")
    logger.info(f"Val score is: {model.score(val_X, val_y)}")
    logger.info(f"Test score is: {model_score}")
    logger.info("Feature importances")
    for f_name, score in sorted(zip(RFC_PARAM.TRAIN_FEATS, model.feature_importances_)):
        logger.info(f"{f_name}, {round(score, 2)}")
    return round(model_score, 3)

if __name__ == '__main__':

    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'
    sample_file = r'd:\ProjectRicequant\fxincome\fxincome_processed.csv'

    PERIOD = '1d_fwd'

    sample_df = pd.read_csv(sample_file, parse_dates=['date'])
    sample_df = sample_df[sample_df['date'] > datetime.datetime(2018, 1, 1)]
    train_X, train_y, val_X, val_y, test_X, test_y = generate_dataset(sample_df, val_ratio=0.1, test_ratio=0.1)

    model, name = random_train(train_X, train_y, val_X, val_y, test_X, test_y)
    joblib.dump(model, f"models/{name}.pkl")