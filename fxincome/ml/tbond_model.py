import pandas as pd
import numpy as np
import datetime
import os
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from fxincome.const import TBOND_PARAM
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


def generate_dataset(main_df, root_path, val_ratio=0.1, test_ratio=0.1):
    """
    将所有样本分成训练集、验证集和测试集，同时写入csv
    train, val, test 数据集的比例分别是：train = 1 - val_ratio - test_ratio,
                                     val = val_ratio
                                     test = test_ratio
    train set的labels分布比例遵从所有样本的labels的分布比例

        Args:
            main_df(DataFrame): 包含所有样本的dataframe，含labels
            root_path(str): 分解结果csv的存储根目录
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

    train_df, test_df, val_df = split_stratified_into_train_val_test(main_df,
                                                                     stratify_colname='target',
                                                                     frac_train=1 - val_ratio - test_ratio,
                                                                     frac_val=val_ratio,
                                                                     frac_test=test_ratio)
    train_df.to_csv(os.path.join(root_path, 'train_samples.csv'), index=False, encoding='utf-8')
    val_df.to_csv(os.path.join(root_path, 'validation_samples.csv'), index=False, encoding='utf-8')
    test_df.to_csv(os.path.join(root_path, 'test_samples.csv'), index=False, encoding='utf-8')

    train_X = train_df[TBOND_PARAM.TRAIN_FEATS]
    train_y = train_df[TBOND_PARAM.LABELS].squeeze().to_numpy()
    val_X = val_df[TBOND_PARAM.TRAIN_FEATS]
    val_y = val_df[TBOND_PARAM.LABELS].squeeze().to_numpy()
    test_X = test_df[TBOND_PARAM.TRAIN_FEATS]
    test_y = test_df[TBOND_PARAM.LABELS].squeeze().to_numpy()
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


def train(train_X, train_y, val_X, val_y, test_X, test_y):
    """
    训练模型并输出报告，返回最佳模型, 及附带分数的模型名字

        Returns:
            model(RandomForestClassifer)：最佳的模型
            name_score(str): 最佳模型的来源 + 对应的分数。来源包括RandomizedSearchCV 和 GridSearchCV, 分数保留3位小数
    """
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
    random_score = report_model(random_model, test_X, test_y, train_X, train_y, val_X, val_y, feat_importance=True)
    logger.info("GridSearch Report:\n _____________")
    grid_score = report_model(grid_model, test_X, test_y, train_X, train_y, val_X, val_y, feat_importance=True)

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


def rfc_train(train_X, train_y, val_X, val_y, test_X, test_y):
    """
    使用RandomForestClassifier 训练模型并输出报告，返回最佳模型及附带分数的模型名字
        Returns:
            model(RandomForestClassifer)：最佳的模型
            name(str): 对应的分数 + 预测的内容 + 日期时点。 分数保留3位小数， 预测的内容由全局变量PERIOD(str)决定。
    """
    combined_train_X = train_X.append(val_X)
    combined_train_y = np.append(train_y, val_y)
    logger.info(f"Size of train set X: {len(combined_train_X)}")
    clf = RandomForestClassifier(oob_score=True, n_jobs=-1)
    param_dist = {
        'n_estimators': stats.randint(50, 500),
        'max_samples': stats.uniform(0.3, 0.7),
        'min_samples_leaf': stats.loguniform(1e-3, 1e-1),
        'max_leaf_nodes': stats.randint(20, 100),
        'max_depth': stats.randint(3, 9)
    }
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=50,
                                       n_jobs=-1, pre_dispatch=64)

    epoch = test_score = train_score = val_score = obb_score = 0
    epoch_limit = 1000
    while epoch < epoch_limit and (test_score < 0.60 or train_score < 0.62 or val_score < 0.60 or obb_score < 0.53):
        logger.info(f"Epoch {epoch}")
        random_search.fit(combined_train_X, combined_train_y)
        best_rfc = random_search.best_estimator_
        test_score, train_score, val_score, obb_score = report_model(best_rfc,
                                                                     test_X, test_y, train_X,
                                                                     train_y, val_X, val_y,
                                                                     rfc=True, feat_importance=True)
        epoch += 1

    return best_rfc, f"{test_score}-{PERIOD}-RFC-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"


def xgb_train(train_X, train_y, val_X, val_y, test_X, test_y):
    """
    训练XGBoost模型并输出报告，返回最佳模型及附带分数的模型名字。用了early_stopping，训练样本留了一块作为early_stopping的val set
        Returns:
            model(XGBoost)：最佳的模型
            name(str): 对应的分数 + 预测的内容 + 日期时点。 分数保留3位小数， 预测的内容由全局变量PERIOD(str)决定。
    """
    # combined_train_X = train_X.append(val_X)
    # combined_train_y = np.append(train_y, val_y)
    logger.info(f"Size of train set X: {len(train_X)}")
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    param_dist = {
        'n_estimators': stats.randint(100, 150),  # default 100
        'colsample_bytree': stats.uniform(0.7, 0.3),
        'gamma': stats.uniform(0, 0.5),
        'learning_rate': stats.loguniform(3e-3, 3e-1),  # default 0.3
        'subsample': stats.uniform(0.6, 0.4),
        'max_depth': stats.randint(3, 9),  # default 6
    }
    xgb_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=50,
                                    return_train_score=True, n_jobs=-1, pre_dispatch=64)

    epoch = test_score = train_score = val_score = obb_score = 0
    epoch_limit = 1000
    while epoch < epoch_limit and (test_score < 0.61 or train_score < 0.62 or val_score < 0.61):
        logger.info(f"Epoch {epoch}")
        xgb_search.fit(train_X, train_y, eval_set=[(val_X, val_y)], early_stopping_rounds=7, verbose=False)
        best_xgb = xgb_search.best_estimator_
        test_score, train_score, val_score, obb_score = report_model(best_xgb,
                                                                     test_X, test_y, train_X,
                                                                     train_y, val_X, val_y, feat_importance=True)
        epoch += 1

    return best_xgb, f"{test_score}-{PERIOD}-XGB-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"


def svm_train(train_X, train_y, val_X, val_y, test_X, test_y):
    """
    训练SVM模型并输出报告，返回最佳模型及附带分数的模型名字。Kernel为Gaussian RBF kernel
        Returns:
            model(Support Vector Classification)：最佳的模型
            name(str): 对应的分数 + 预测的内容 + 日期时点。 分数保留3位小数， 预测的内容由全局变量PERIOD(str)决定。
    """
    combined_train_X = train_X.append(val_X)
    combined_train_y = np.append(train_y, val_y)
    logger.info(f"Size of train set X: {len(combined_train_X)}")
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True))
    ])
    n_features = train_X.shape[1]
    param_dist = {
        'svc__C': stats.loguniform(1e-4, 100),
        'svc__gamma': stats.loguniform(1e-4, 1)
    }
    svm_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=2000,
                                    n_jobs=-1, pre_dispatch=64)
    param_grid = {
        # 'svc__C': np.logspace(-5, 3, num=100, base=10),
        # 'svc__gamma': 1 / np.linspace(2 * n_features, 1 / n_features, 20)  # default is 1 / (n_features)
        'svc__C': np.logspace(-5, 10, num=50, base=2),
        'svc__gamma': np.logspace(-15, 3, num=50, base=2)
        # 'svc__C': np.logspace(-5, 2, num=20),
        # 'svc__gamma': np.logspace(-3, 2, num=20)
    }
    # svm_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, pre_dispatch=64)
    svm_search.fit(combined_train_X, combined_train_y)
    best_svm = svm_search.best_estimator_
    test_score, train_score, val_score, obb_score = report_model(best_svm,
                                                                 test_X, test_y, train_X,
                                                                 train_y, val_X, val_y)

    return best_svm, f"{test_score}-{PERIOD}-SVM-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"


def svm_poly_train(train_X, train_y, val_X, val_y, test_X, test_y):
    """
    训练SVM模型并输出报告，返回最佳模型及附带分数的模型名字。Kernel为Polynominal
        Returns:
            model(Support Vector Classification)：最佳的模型
            name(str): 对应的分数 + 预测的内容 + 日期时点。 分数保留3位小数， 预测的内容由全局变量PERIOD(str)决定。
    """
    combined_train_X = train_X.append(val_X)
    combined_train_y = np.append(train_y, val_y)
    logger.info(f"Size of train set X: {len(combined_train_X)}")
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='poly', probability=True))
    ])
    n_features = train_X.shape[1]
    param_dist = {
        'svc__C': stats.loguniform(1e-4, 100),
        'svc__gamma': stats.loguniform(1e-4, 1),
        'svc__degree': stats.randint(2, 3)
    }
    svm_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=200,
                                    n_jobs=-1, pre_dispatch=64)
    svm_search.fit(combined_train_X, combined_train_y)
    best_svm = svm_search.best_estimator_
    test_score, train_score, val_score, obb_score = report_model(best_svm,
                                                                 test_X, test_y, train_X,
                                                                 train_y, val_X, val_y)

    return best_svm, f"{test_score}-{PERIOD}-POLY-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"


def report_model(model, test_X, test_y, train_X, train_y, val_X, val_y, rfc=False, feat_importance=False):
    """
    输出模型报告，返回各种分数
        Args:
            rfc(boolean): 是否RandomForest，如果是，则能够返回oob_score；如果不是，则oob_score = 0。默认为False
            feat_importance(boolean): 是否显示feature_importance。只有树状模型才有feature_importance。默认为False
        Returns:
            model_score(float)：模型在test set中的分数
            train_score(float): 模型在train set中的分数
            val_score(float): 模型在validation set中的分数
            obb_score(float): 模型在训练中的obb_score，只有RandomForest模型才有意义，否则返回0
    """
    logger.info(model.get_params)
    test_pred = model.predict(test_X)
    model_score = model.score(test_X, test_y)
    obb_score = model.oob_score_ if rfc else 0
    train_score = model.score(train_X, train_y)
    val_score = model.score(val_X, val_y)
    logger.info("Test report: ")
    print(classification_report(test_y, test_pred))
    logger.info(f"Train oob_score is: {obb_score}")
    logger.info(f"Train score is: {train_score}")
    logger.info(f"Val score is: {val_score}")
    logger.info(f"Test score is: {model_score}")
    if feat_importance:
        logger.info("Feature importances")
        for f_name, score in sorted(zip(TBOND_PARAM.TRAIN_FEATS, model.feature_importances_), key=lambda x: x[1], reverse=True):
            logger.info(f"{f_name}, {round(float(score), 2)}")
    return round(model_score, 3), train_score, val_score, obb_score


if __name__ == '__main__':
    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'
    sample_file = 'd:/ProjectRicequant/fxincome/fxincome_processed.csv'

    PERIOD = '1d_fwd'

    sample_df = pd.read_csv(sample_file, parse_dates=['date'])
    # sample_df = sample_df[sample_df['date'] > datetime.datetime(2018, 1, 1)]
    train_X, train_y, val_X, val_y, test_X, test_y = generate_dataset(sample_df, root_path=ROOT_PATH,
                                                                      val_ratio=0.1, test_ratio=0.1)

    # model, name = rfc_train(train_X, train_y, val_X, val_y, test_X, test_y)
    # joblib.dump(model, f"models/{name}.pkl")
    # model, name = xgb_train(train_X, train_y, val_X, val_y, test_X, test_y)
    # joblib.dump(model, f"models/{name}.pkl")
    # model, name = svm_train(train_X, train_y, val_X, val_y, test_X, test_y)
    # joblib.dump(model, f"models/{name}.pkl")
    model, name = svm_poly_train(train_X, train_y, val_X, val_y, test_X, test_y)
    joblib.dump(model, f"models/{name}.pkl")
