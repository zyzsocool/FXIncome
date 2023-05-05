# coding: utf-8
import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from fxincome import logger
from fxincome.const import PATH, SPREAD
from fxincome.spread import preprocess_data
from fxincome.utils import ModelAttr, JsonModel
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import log_loss, classification_report
import lightgbm as lgbm
from lightgbm import early_stopping
import optuna
from optuna.samplers import TPESampler, GridSampler, RandomSampler
from optuna.integration import LightGBMPruningCallback


def split_stratify_train(data: pd.DataFrame, label_ratio_low: float, label_ratio_high: float, test_size=0.2):
    """
    Split data into train and test set.
    Train set is split with stratify label ratio between label_ratio_low and label_ratio_high.
    Args:
        data(Dataframe): Dataframe to be split. Must contain a column named 'LABEL'.
        label_ratio_low(float): Lower bound of label ratio in train set.
        label_ratio_high(float: Upper bound of label ratio in train set.
        test_size: Ratio of test set.
    Returns:
        X_train(Dataframe): Train set features.
        X_test(Dataframe): Test set features.
        y_train(Dataframe): Train set labels.
        y_test(Dataframe): Test set labels.
    """
    while True:
        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['LABEL']), data['LABEL'],
                                                            test_size=test_size)
        if (y_train.sum() / len(y_train) >= label_ratio_low) and (y_train.sum() / len(y_train) <= label_ratio_high):
            break
    logger.info(f'Label 1 ratio of train set after split:{y_train.sum() / len(y_train)}')
    return X_train, X_test, y_train, y_test


def plot_graph(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, model):
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

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


def generate_dataset(days_back: int, n_samples: int, days_forward: int, spread_threshold: float, features: list[str],
                     last_n_for_test: int = 3):
    """
    Generate dataset for training and testing. Training set is yet to be split into train and validation set.
    Args:
        days_back (int): number of trade days backward for features,
                         when features of these past days are included to this sample.
                         Features are like yields, spreads, volumns, outstanding balances ...
                         for leg1 and leg2 on each past day.
        n_samples (int): number of samples selected from data. Samples are selected since leg2's IPO day.
                         Yields and other features before leg2's IPO date are doubtful.
                         To calculate features of previous days, we begin from leg2's IPO date + days_back
                         Last sample = leg2's IPO date + days_back + n_samples - 1
                         Only trading days are counted. So it's different from calendar days.
        days_forward (int): number of trade days forward for labels. It's NOT calendar day.
        spread_threshold (float): spread threshold for labels. The unit is percent point, eg: 0.01 is 0.01%
                        Its sign determines whether spread is wider or narrower.
                        If  POSITIVE, assuming spread is wider, then:
                            during the period between T and T + days_forward,
                            if any day's spread - spread_T > spread_threshold, then label = 1.
                        If NEGATIVE, assuming spread is narrower, then:
                            during the period between T and T + days_forward,
                            if any day's spread - spread_T < spread_threshold, then label = 1.
        features (list[str]): Only these features are included in the final dataframe.
                              Note that labels are not included in features.
        last_n_for_test (int): Last n bonds are used for testing. It must be >= 2
                        Default is 3, which means 2 pairs of bonds are used for training.
    Returns:
        train_X(DataFrame): DataFrame for training features.
        train_Y(DataFrame): 1D DataFrame for training labels, 0 or 1.
        test_X(DataFrame): DataFrame for testing features.
        test_Y(DataFrame): 1D DataFrame for testing labels, 0 or 1.
    """
    assert last_n_for_test >= 2, 'test_num must be >= 2'
    # train and validation set, from 180210_190205 to 220210_220215
    df_train_val = preprocess_data.feature_engineering(SPREAD.CDB_CODES[0], SPREAD.CDB_CODES[1], days_back, n_samples,
                                                       days_forward, spread_threshold, features)
    for i in range(1, len(SPREAD.CDB_CODES) - last_n_for_test):
        df_train_val = pd.concat([df_train_val,
                                  preprocess_data.feature_engineering(SPREAD.CDB_CODES[i], SPREAD.CDB_CODES[i + 1],
                                                                      days_back, n_samples,
                                                                      days_forward, spread_threshold,
                                                                      features)], axis=0)

    # test set, from 220215_220210 to 220220_230205
    test_leg1_index = len(SPREAD.CDB_CODES) - last_n_for_test
    test_leg2_index = len(SPREAD.CDB_CODES) - last_n_for_test + 1
    df_test = preprocess_data.feature_engineering(SPREAD.CDB_CODES[test_leg1_index], SPREAD.CDB_CODES[test_leg2_index],
                                                  days_back, n_samples,
                                                  days_forward, spread_threshold, features)
    for i in range(test_leg2_index, len(SPREAD.CDB_CODES) - 1):
        df_test = pd.concat([df_test, preprocess_data.feature_engineering(SPREAD.CDB_CODES[i], SPREAD.CDB_CODES[i + 1],
                                                                          days_back, n_samples,
                                                                          days_forward, spread_threshold, features)],
                            axis=0)

    logger.info(f"Number of rows with null values in train and validation set: {df_train_val.isna().any(axis=1).sum()}")
    logger.info(f"Number of rows with null values in test set: {df_test.isna().any(axis=1).sum()}")
    df_train_val = df_train_val.dropna()
    df_test = df_test.dropna()
    logger.info(f"Test samples: {len(df_test)}, Test ratio: {len(df_test) / (len(df_train_val) + len(df_test)):.4f}")
    df_train_val.to_csv(PATH.SPREAD_DATA + 'train_samples.csv', index=False, encoding='utf-8')
    df_test.to_csv(PATH.SPREAD_DATA + 'test_samples.csv', index=False, encoding='utf-8')
    train_X = df_train_val.drop(columns=['LABEL'])
    train_Y = df_train_val['LABEL']
    test_X = df_test.drop(columns=['LABEL'])
    test_Y = df_test['LABEL']
    return train_X, train_Y, test_X, test_Y


def xgb_scikit_random_train(train_X, train_Y, test_X, test_Y):
    """
    Train XGBoost model using scikit-learn RandomizedSearchCV, and output report.
    Train set is split into train and validation set.
    The validation set is used for early stopping.
    Returns:
        model(XGBClassifier): best model
        name(str): {score}_XGB_{datetime}.
    """

    x_train, x_val, y_train, y_val = train_test_split(train_X, train_Y, test_size=0.1)
    logger.info(f"Train set size: {len(x_train)}, validation set(for early stopping) size: {len(x_val)}")
    objective = 'binary:logistic'
    eval_metric = 'logloss'
    early_stopping_rounds = 7
    n_iter = 100  # number of iterations for RandomizedSearchCV
    param_dist = {
        'n_estimators': stats.randint(100, 300),  # default 100, try 100-300
        'max_depth': stats.randint(5, 10),  # default 6, try 5-10
        'gamma': stats.uniform(0, 10),  # default 0, try 0-10
        'subsample': stats.uniform(0.8, 0.2),  # default 1, try 0.8-1
        'colsample_bytree': stats.uniform(0.7, 0.3),  # default 1, try 0.7-1
        'learning_rate': stats.loguniform(1e-3, 10),  # default 0.3, try 0.001-10
    }
    clf = xgb.XGBClassifier(objective=objective, eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds)
    xgb_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter,
                                    return_train_score=True, n_jobs=-1, pre_dispatch=64)
    xgb_search.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    model = xgb_search.best_estimator_
    test_score, train_score, val_score = report_model(model, test_X, test_Y, x_train, y_train, x_val, y_val)

    return model, f"spread_{test_score}_XGB_{datetime.datetime.now():%Y%m%d_%H%M}"


def lr_train(train_X, train_Y, test_X, test_Y):
    logger.info(f"Train size: {len(train_X)}, Test size: {len(test_X)}")
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(solver='liblinear', max_iter=500))
    ])
    param_dist = {
        'logistic__C': stats.loguniform(1e-4, 10),
        'logistic__penalty': ['l1', 'l2'],
    }
    lr_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=1000, n_jobs=-1, pre_dispatch=64)
    lr_search.fit(train_X, train_Y)
    model = lr_search.best_estimator_
    x_train, x_val, y_train, y_val = train_test_split(train_X, train_Y, test_size=0.1)
    test_score, train_score, val_score = report_model(model, test_X, test_Y, x_train, y_train, x_val, y_val)
    return model, f"spread_{test_score}_LR_{datetime.datetime.now():%Y%m%d_%H%M}"


class OptunaCVObjective(object):
    early_stopping_rounds = 10

    def __init__(self, x_df, y_df):
        self.x_df = x_df
        self.y_df = y_df

    def __call__(self, trial):
        search_param = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 16, 31),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 1),
            "subsample": trial.suggest_float("subsample", 0.8, 0.99),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.99),
        }
        fixed_param = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 1000),
            "subsample_freq": trial.suggest_int('subsample_freq', 1, 1),
        }
        param_grid = {**search_param, **fixed_param}

        cv = StratifiedKFold(n_splits=5, shuffle=True)

        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(self.x_df, self.y_df)):
            x_train, x_test = self.x_df.iloc[train_idx], self.x_df.iloc[test_idx]
            y_train, y_test = self.y_df.iloc[train_idx], self.y_df.iloc[test_idx]

            model = lgbm.LGBMClassifier(objective="binary", verbose=-1, **param_grid)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_test, y_test)],
                eval_metric="binary_logloss",
                callbacks=[early_stopping(stopping_rounds=OptunaCVObjective.early_stopping_rounds)],
            )
            preds = model.predict_proba(x_test)
            cv_scores[idx] = log_loss(y_test, preds)
        #  Optimization objective is to minimize the mean of cv scores.
        return np.mean(cv_scores)


def lgb_train(X_train, Y_train, X_test, Y_test):
    study = optuna.create_study(direction="minimize", study_name="StudyLGBMClassifier")
    objective = OptunaCVObjective(X_train, Y_train)
    # Find the best hyperparameters by cross-validation.
    study.optimize(objective, n_trials=100)
    # x_train, y_train are split from X_train, Y_train. x_val, y_val are for early stopping.
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)
    # Train the model one last time with the best hyperparameters.
    model = lgbm.LGBMClassifier(objective="binary", verbose=-1, **study.best_params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(stopping_rounds=OptunaCVObjective.early_stopping_rounds)]
    )
    test_score, train_score, val_score = report_model(model, X_test, Y_test, x_train, y_train, x_val, y_val)
    return model, f"spread_{test_score}_LGB_{datetime.datetime.now():%Y%m%d_%H%M}"


def report_model(model, test_X, test_Y, train_X, train_Y, val_X, val_Y):
    params = ['n_estimators', 'max_depth', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 'alpha',
              'lambda', 'learning_rate', 'scale_pos_weight', 'booster', 'objective', 'eval_metric',
              'early_stopping_rounds'] + \
             ['logistic__C', 'logistic__penalty', 'logistic__class_weight'] + \
             ['min_child_samples', 'min_child_weight', 'num_leaves', 'lambda_l1', 'lambda_l2', 'min_split_gain',
              'subsample_freq']
    #  model.get_params() returns a dict of all parameters. Only print the ones we care about.
    best_params = [f'{param}: {value}' for param, value in model.get_params().items() if param in params]
    if hasattr(model, 'get_booster'):
        best_params.append(f'num_boosted_rounds: {model.get_booster().num_boosted_rounds()}')
    elif hasattr(model, 'booster_'):
        best_params.append(f'num_trees: {model.booster_.num_trees()}')
    logger.info("Model parameters: ")
    logger.info(best_params)
    #  Train、val、test scores
    test_pred = model.predict(test_X)
    model_score = model.score(test_X, test_Y)
    train_score = model.score(train_X, train_Y)
    val_score = model.score(val_X, val_Y)
    logger.info("Test report: ")
    print(classification_report(test_Y, test_pred))
    logger.info(f"Train score is: {train_score:.4f}, Val score is: {val_score:.4f}, Test score is: {model_score:.4f}")
    #  Feature importances
    if hasattr(model, 'feature_importances_'):
        logger.info("Feature importances")
        for f_name, score in sorted(zip(test_X.columns, model.feature_importances_), key=lambda x: x[1],
                                    reverse=True):
            logger.info(f"{f_name}, {float(score):.2f}")
    plot_graph(train_X, train_Y, test_X, test_Y, model)
    return round(model_score, 3), train_score, val_score


def save_model_attr(name: str, days_back: int, n_samples: int, days_forward: int, spread_threshold: float,
                    features: list,
                    last_n_bonds_for_test: int):
    model_attr = ModelAttr(
        name=name,
        features=features,
        labels={
            'LABEL': {
                'value_scope': [0, 1],
                'days_forward': days_forward,
                'spread_threshold': spread_threshold,
            }
        },
        other={
            'days_back': days_back,
            'n_samples': n_samples,
            'last_n_bonds_for_test': last_n_bonds_for_test,
            'bonds': SPREAD.CDB_CODES
        }
    )
    JsonModel.save_attr(model_attr, PATH.SPREAD_MODEL + JsonModel.JSON_NAME)


def main():
    days_back = 5
    n_samples = 200
    days_forward = 10
    spread_threshold = 0.01
    last_n_bonds_for_test = 3
    features = preprocess_data.select_features(days_back=days_back)

    train_X, train_Y, test_X, test_Y = generate_dataset(days_back, n_samples, days_forward, spread_threshold,
                                                        features, last_n_bonds_for_test)

    # model, name = lgb_train(train_X, train_Y, test_X, test_Y)
    # joblib.dump(model, PATH.SPREAD_MODEL + f'{name}.pkl')

    # model, name = xgb_scikit_random_train(train_X, train_Y, test_X, test_Y)
    # model.save_model(PATH.SPREAD_MODEL + f'{name}.json')
    # model.get_booster().dump_model(PATH.SPREAD_MODEL + f'{name}_trees.txt')
    # config = model.get_booster().save_config()
    #
    # with open(PATH.SPREAD_MODEL + f'{name}_config.json', 'w') as f:
    #     json.dump(config, f)

    model, name = lr_train(train_X, train_Y, test_X, test_Y)
    joblib.dump(model, PATH.SPREAD_MODEL + f'{name}.pkl')
    save_model_attr(name, days_back, n_samples, days_forward, spread_threshold, features, last_n_bonds_for_test)


if __name__ == '__main__':
    main()
