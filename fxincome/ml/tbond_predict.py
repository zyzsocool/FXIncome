import pandas as pd
import numpy as np
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable Tensorflow messages < ERROR
import joblib
import matplotlib.pyplot as plt
import sklearn as sk
import xgboost
from fxincome.const import TBOND_PARAM, PATH
from fxincome.ml import tbond_process_data, tbond_model, tbond_nn_predata
from fxincome import logger
from fxincome.utils import JsonModel, ModelAttr
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier


def show_tree(model):
    """
    展示决策树模型中的任意一颗树，如果是XGBClassifier，还将pot_importance
        Args:
            model: 决策树模型，可以为RandomForestClassifier 或 XGBClassifier
        Returns:
            none
    """
    plt.rcParams.update({'figure.figsize': (20, 16)})
    plt.rcParams.update({'font.size': 12})
    if isinstance(model, RandomForestClassifier):
        tree = model.estimators_[0].tree_
        logger.info(f"Tree depth: {tree.max_depth}")
        sk.tree.plot_tree(tree, feature_names=TBOND_PARAM.TRAIN_FEATS, filled=True)

    elif isinstance(model, xgboost.XGBClassifier):
        logger.info(f"Tree depth: {model.max_depth}")
        xgboost.plot_tree(model, num_trees=12)
        xgboost.plot_importance(model, importance_type='weight')
    else:
        raise NotImplementedError("Unknown Tree Model!")
    plt.show()


def show_prediction(name, today, pred, proba, label_type, future_period):
    """
    辅助函数，展示预测结果。
        Args:
            pred(float): 0为下跌，1为上涨
            proba(list): [下跌概率， 上涨概率]
    """
    if label_type == 'fwd':
        logger.info(f"{name} - {today} - 16国债19 ytm在{future_period}个交易日后将{'上涨' if pred == 1 else '下跌'}")
    elif label_type == 'avg':
        logger.info(f"{name} - {today} - 16国债19 ytm在未来{future_period}个交易日的均值对比今天将{'上涨' if pred == 1 else '下跌'}")
    else:
        raise NotImplementedError("Unknown label_type")

    logger.info(f"{name} - 预测下跌概率：{float(proba[0]):.2f}，预测上涨概率：{float(proba[1]):.2f}")


def vote(row, plain_names: list, nn_names: list, mode: str = 'hard', weights: list = []):
    """
    辅助函数，用于对每一行进行Ensemble vote。投票模式可分为hard和soft两种形式。可设置各个模型的投票权重，具体算法见参数说明。
        Args:
            plain_names(list): A list of strs. 传统模型的名字。
            nn_names(list): A list of strs. 神经网络模型的名字。
            mode(str): 投票方式，['hard', 'soft']。'hard'是预测方向(0,1), 'soft'是预测上涨可能性。
                模型预测的方向在row中的column name以'_pred'结尾
                模型预测的上涨可能性row中的column name以'_up'结尾
            weights(list): A list of floats. 每个模型的投票权重，顺序与plain_names和nn_names一致。默认所有权重为1.0，即[1，1，1...]
                设某模型的权重为w，其投票分值为x，则经权重调整后的分值
                y = 0.5 + (x - 0.5) * w
        Returns:
            Ensemble的预测结果，0 or 1
    """
    names = plain_names + nn_names
    if len(weights) == 0:
        weights = [1.0] * (len(names))  # 默认权重全部为1.0
    if len(weights) != len(names):
        raise ValueError(f"Len of weights({len(weights)} not equal to len of models({len(names)})")
    score = 0
    threshold = len(names) / 2  # 过半数的阈值，<= 阈值 返回 0；> 阈值 返回 1
    for i, name in enumerate(names):
        w = weights[i]
        if mode == 'hard':
            score = score + 0.5 + (row[name + '_pred'] - 0.5) * w
        elif mode == 'soft':
            score = score + 0.5 + (row[name + '_up'] - 0.5) * w
        else:
            raise NotImplementedError("Unknown weight type")
    if score <= threshold:
        return 0
    else:
        return 1


def eval_plain_models(names: list, engineered_df):
    """
    用最新的数据检验模型。这些模型通常是树模型、SVM、LR等传统模型。
    输入可为多个模型。对于每个模型，依次显示模型的name， params, test_report, score, feature_importances（如有）
    只有树状模型才显示feature_importance，目前只支持'XGBClassifier'和'RandomForestClassifier'
        Args:
            names(List): A list of model names. Models需要满足以下条件：
                1.具有以下methods: predict(X), predict_proba(X), score(X,y)
                2.X是dataframe的单行。
            engineered_df(DataFrame): 检验数据，含日期，含labels，需要做好预处理
        Returns:
            history_result(DataFrame): 'date', 'actual', [每个model预测的'result', 'pred', 'actual', 'down_proba', 'up_proba']
    """
    models = JsonModel.load_plain_models(names, PATH.YTM_MODEL, 'joblib')
    df = engineered_df[['date']]
    names = []
    col_names = ['date']
    for attr, model in models.items():
        name = attr.name
        logger.info(f"Model: {name}")
        logger.info(model.get_params)
        if 'xgb' in name or 'rfc' in name:
            logger.info("Feature importances")
            for f_name, score in sorted(zip(attr.features, model.feature_importances_), key=lambda x: x[1],
                                        reverse=True):
                logger.info(f"{f_name}, {float(score):.2f}")
        X = engineered_df[attr.features]
        y = engineered_df[attr.labels].squeeze().to_numpy()
        test_pred = model.predict(X)
        model_score = model.score(X, y)
        print(classification_report(y, test_pred))
        logger.info(f"Test score is: {model_score}")

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
    return history_result


def eval_models(plain_names: list, nn_names: list, df, seq_len, weights: list = []):
    """
    用最新的数据检验模型，并综合这些模型形成Ensemble，以投票方式决定预测结果。
    投票方式有两种：['soft', 'hard']。'hard'是预测方向(0,1), 'soft'是预测上涨可能性。
    这些模型分为plain models（通常是树模型、SVM、LR等传统模型）和nn models（通常是神经网络模型）
    输入为模型名字的列表。
    对于plain models，依次显示模型的name， params, test_report, score, features, feature_importances（如有）
        只有树状模型才显示feature_importance，目前只支持'XGBClassifier'和'RandomForestClassifier'
    对于nn models，依次显示模型的name，features，summary（结构）
        Args:
            plain_names(list): A list of strs. 传统模型的名字。模型需要满足以下条件：
                1.具有以下methods: predict(X), predict_proba(X), score(X,y)
                2.X是dataframe的单行。
            nn_names(list): A list of strs. 神经网络模型的名字。
            df(DataFrame): 检验数据，含日期（date），需预处理好所有features和labels，输入的模型应基于这些features和labels训练。
            seq_len(int): time steps的长度
            weights(list): A list of floats. 每个模型的投票权重，顺序与plain_names和nn_names一致。默认所有权重为1.0，即[1，1，1...]
                设某模型的权重为w，其投票分值为x，则经权重调整后的分值为：
                y = 0.5 + (x - 0.5) * w
        Returns:
            history_result(DataFrame): 结构如下
            'date'(index), 'hard_pred', 'soft_pred', 'actual',
            [每个model预测的'result', 'pred', 'actual', 'down_proba', 'up_proba']
    """
    plain_result = eval_plain_models(plain_names, df)
    nn_models = JsonModel.load_nn_models(nn_names)  # return a dict(key: ModelAttr, value: model)
    dates = df[df.date.isin(plain_result.date)].date.values  # dates on both plain models and nn models.
    dates = dates[seq_len:]  # 前seq_len个样本用于预测
    result = plain_result[plain_result.date.isin(dates)].copy()  # 只记录有预测的部分
    logger.info(f"plain sample size: {len(plain_result)}")
    logger.info(f"nn sample size: {len(result)}")

    names = []
    for attr, model in nn_models.items():
        name = attr.name
        names.append(name)
        scaled_df, scale_stats = tbond_nn_predata.scale(df, attr.scaled_feats, attr.stats)
        x = []
        for today in dates:
            x.append(tbond_nn_predata.gen_pred_x(scaled_df, today, attr.features, seq_len=seq_len))
        x = np.array(x)
        y = df[attr.labels].squeeze().to_numpy()
        y = y[seq_len:]  # 与x对应，从第seq_len个日期开始预测
        model_score = model.evaluate(x, y, verbose=0)
        logger.info(f"Model: {name}  Accuracy is: {model_score[1]:.4f} Loss is: {model_score[0]:.4f}")
        probs = model.predict(x).flatten()  # change shape: (n, 1) to shape(n,)
        preds = (probs > 0.5).astype(int)  # change probability to direction. prob <= 0.5 is 0, > 0.5 is 1
        result.insert(len(result.columns), column=f'{name}_pred', value=preds)
        result.insert(len(result.columns), column=f'{name}_actual', value=y)
        result[name] = result.apply(lambda x: 'Right' if x[f'{name}_pred'] == x[f'{name}_actual'] else 'Wrong', axis=1)
        result[f'{name}_down'], result[f'{name}_up'] = 1 - probs, probs
    #  Display the accuracy of models
    for name in plain_names:
        hit = result.apply(lambda x: 1 if x[name] == 'Right' else 0, axis=1).sum()
        acc = hit / len(result)
        logger.info(f"Model {name} Accuracy: {acc:.4f}")
    #  Ensemble voting
    result['hard_pred'] = result.apply(lambda row: vote(row, plain_names, nn_names, mode='hard', weights=weights),
                                       axis=1)
    result['soft_pred'] = result.apply(lambda row: vote(row, plain_names, nn_names, mode='soft', weights=weights),
                                       axis=1)
    result = result.set_index('date')
    plain_result = plain_result.set_index('date')
    result['actual'] = plain_result[plain_names[0] + '_actual']
    result['hard'] = result.apply(lambda x: 'Right' if x.hard_pred == x.actual else 'Wrong', axis=1)
    result['soft'] = result.apply(lambda x: 'Right' if x.soft_pred == x.actual else 'Wrong', axis=1)
    hard_hit = result.apply(lambda x: 1 if x.hard_pred == x.actual else 0, axis=1).sum()
    soft_hit = result.apply(lambda x: 1 if x.soft_pred == x.actual else 0, axis=1).sum()
    hard_acc = hard_hit / len(result)
    soft_acc = soft_hit / len(result)
    logger.info(f"Hard Accuracy:{hard_acc:.4f}  Soft Accuracy:{soft_acc:.4f}")

    #  Reorder the column names
    col_names = plain_result.columns.tolist()
    for name in names:
        col_names.append(name)
        col_names.append(f'{name}_pred')
        col_names.append(f'{name}_actual')
        col_names.append(f'{name}_down')
        col_names.append(f'{name}_up')
    col_names.insert(0, 'hard')
    col_names.insert(1, 'soft')
    col_names.insert(2, 'hard_pred')
    col_names.insert(3, 'soft_pred')
    col_names.insert(4, 'actual')
    history_result = result[col_names].copy()

    return history_result


def ensemble_pred(plain_names: list, nn_names: list, df, seq_len, weights: list = [], future_period=1, label_type='fwd'):
    """
    用最新的数据检验模型，
        Args:
            plain_names(list): A list of strs. 传统模型的名字。模型需要满足以下条件：
                1.具有以下methods: predict(X), predict_proba(X)
                2.X是dataframe的单行。
            nn_names(list): A list of strs. 神经网络模型的名字。
            df(DataFrame): 原始数据，无需做预处理。
            seq_len(int): time steps的长度
            weights(list): A list of floats. 每个模型的投票权重，顺序与plain_names和nn_names一致。默认所有权重为1.0，即[1，1，1...]
                设某模型的权重为w，其投票分值为x，则经权重调整后的分值为：
                y = 0.5 + (x - 0.5) * w
            future_period(int): label的观察期，用于对比当日的收盘价，生成涨跌label。
            label_type(str): 预测规则，只限于'fwd'或'avg'，默认为'fwd'
                'fwd': 预测未来第n天对比当日收盘价的涨跌
                'avg': 预测未来n天平均值对比当日收盘价的涨跌
        Returns:
            preds(list): 1D array-like of shape(n_models), model顺序与输入的顺序一样；0为下跌，1为上涨
            probas(list): 2D array-like of shape(n_models, 2), model顺序与输入的顺序一样；最后一项为[下跌概率， 上涨概率]
            hard_pred(int): hard ensemble 预测的涨跌方向
            soft_pred(int): soft ensemble 预测的涨跌方向
    """
    df = tbond_process_data.feature_engineering(df,
                                                select_features=TBOND_PARAM.ALL_FEATS,
                                                future_period=future_period,
                                                label_type=label_type,
                                                dropna=False)
    df = df.drop(['future', 'target'], axis=1)
    today = df.date.iloc[-1]  # today is pandas.Timestamp
    plain_models = JsonModel.load_plain_models(plain_names, PATH.YTM_MODEL, 'joblib')
    nn_models = JsonModel.load_nn_models(nn_names)
    preds = []
    probas = []
    for attr, model in plain_models.items():
        x = df[attr.features].tail(1)
        pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0]
        preds.append(pred)
        probas.append(proba)
        show_prediction(attr.name, today, pred, proba, label_type, future_period)
    for attr, model in nn_models.items():
        scaled_df, scale_stats = tbond_nn_predata.scale(df, attr.scaled_feats, attr.stats)
        x = tbond_nn_predata.gen_pred_x(scaled_df, today, attr.features, seq_len=seq_len)
        X = np.array([x])
        probability = model.predict(X).flatten()[0]  # change shape: (1, 1) to a float
        pred = 1 if probability > 0.5 else 0  # change probability to direction. prob <= 0.5 is 0, > 0.5 is 1
        preds.append(pred)
        probas.append([1 - probability, probability])
        show_prediction(attr.name, today, preds[-1], probas[-1], label_type, future_period)

    model_num = len(plain_names + nn_names)
    if len(weights) == 0:
        weights = [1.0] * model_num  # 默认权重全部为1.0
    if len(weights) != model_num:
        raise ValueError(f"Len of weights({len(weights)} not equal to len of models({model_num})")
    hard_score = 0
    soft_score = 0
    threshold = model_num / 2  # 过半数的阈值，<= 阈值 返回 0；> 阈值 返回 1
    for i, w in enumerate(weights):
        hard_score = hard_score + 0.5 + (preds[i] - 0.5) * w
        soft_score = soft_score + 0.5 + (probas[i][1] - 0.5) * w

    hard_pred = 0 if hard_score <= threshold else 1
    soft_pred = 0 if soft_score <= threshold else 1
    logger.info(f"Hard Ensemble - {today} - 16国债19 ytm在{future_period}个交易日后将{'上涨' if hard_pred == 1 else '下跌'}")
    logger.info(f"Soft Ensemble - {today} - 16国债19 ytm在{future_period}个交易日后将{'上涨' if soft_pred == 1 else '下跌'}")
    return preds, probas, hard_pred, soft_pred


def pred_future(models, df, future_period=1, label_type='fwd'):
    """
    用最新的数据检验模型，
        Args:
            df(DataFrame): 原始数据，无需做预处理
            models(list): A list of models. 每个model必须具有以下methods: predict(X), predict_proba(X)
            future_period(int): label的观察期，用于对比当日的收盘价，生成涨跌label。
            label_type(str): 预测规则，只限于'fwd'或'avg'，默认为'fwd'
                'fwd': 预测未来第n天对比当日收盘价的涨跌
                'avg': 预测未来n天平均值对比当日收盘价的涨跌
        Returns:
            preds(list): 2D array-like of shape(n_models, n_dates), model顺序与输入的顺序一样；0为下跌，1为上涨
            probas(list): 3D array-like of shape(n_models, n_dates, 2), model顺序与输入的顺序一样；最后一项为[下跌概率， 上涨概率]
    """
    df = tbond_process_data.feature_engineering(df,
                                                select_features=TBOND_PARAM.ALL_FEATS,
                                                future_period=future_period,
                                                label_type=label_type,
                                                dropna=False)
    today = df.date.iloc[-1].date()
    last_x = df[TBOND_PARAM.TRAIN_FEATS].tail(1)
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
                logger.info(f"{name} - {today} - 16国债19 ytm在{future_period}个交易日后将下跌")
            elif label_type == 'avg':
                logger.info(f"{name} - {today} - 16国债19 ytm在未来{future_period}个交易日的均值对比今天将下跌")
            else:
                raise NotImplementedError("Unknown label_type")
        elif pred[0] == 1:
            if label_type == 'fwd':
                logger.info(f"{name} - {today} - 16国债19 ytm在{future_period}个交易日后将上涨")
            elif label_type == 'avg':
                logger.info(f"{name} - {today} - 16国债19 ytm在未来{future_period}个交易日的均值对比今天将上涨")
            else:
                raise NotImplementedError("Unknown label_type")
        else:
            logger.info(f"Unknown result: {name} - {pred[0]}")

        logger.info(f"预测下跌概率：{float(proba[0][0]):.2f}，预测上涨概率：{float(proba[0][1]):.2f}")
        preds.append(pred)
        probas.append(proba)
    return preds, probas


def val_models(models, df):
    X = df[TBOND_PARAM.TRAIN_FEATS]
    y = df[TBOND_PARAM.LABELS].squeeze().to_numpy()
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
            for f_name, score in sorted(zip(TBOND_PARAM.TRAIN_FEATS, model.feature_importances_), key=lambda x: x[1],
                                        reverse=True):
                logger.info(f"{f_name}, {float(score):.2f}")
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


def main():
    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'

    sample_file = r'd:\ProjectRicequant\fxincome\fxincome_features_latest.csv'
    sample_df = pd.read_csv(sample_file, parse_dates=['date'])
    test_df = tbond_process_data.feature_engineering(sample_df,
                                                     select_features=TBOND_PARAM.ALL_FEATS,
                                                     future_period=1,
                                                     label_type='fwd')
    test_df.to_csv(os.path.join(ROOT_PATH, 'test_df.csv'), index=False, encoding='utf-8')
    train_X, train_y, val_X, val_y, test_X, test_y = tbond_model.generate_dataset(test_df, root_path=ROOT_PATH,
                                                                                  val_ratio=0.1, test_ratio=0.1)
    # svm_model = joblib.load(f"models/0.626-1d_fwd-XGB-20210618-1433-v2016.pkl")
    xgb_model = joblib.load(f"models/0.626-1d_fwd-XGB-20210618-1454-v2016.pkl")
    rfc_model = joblib.load(f"models/0.605-1d_fwd-RFC-20210619-1346-v2018.pkl")
    # pol_model = joblib.load(f"models/0.626-1d_fwd-XGB-20210618-1454-v2016.pkl")
    plain_models = ['0.626-1d_fwd-XGB-20210618-1454-v2016.pkl', '0.605-1d_fwd-RFC-20210619-1346-v2018.pkl']
    # nn_models = ['Checkpoint-10-SEQ-1-PRED-20210903-1639.model']
    nn_models = []
    vote_model = EnsembleVoteClassifier(clfs=[xgb_model, rfc_model],
                                        weights=[1, 1], voting='soft', fit_base_estimators=False)
    vote_model.fit(val_X, val_y)
    # history_result = val_models([vote_model, xgb_model, rfc_model], test_df)
    # pred_future([vote_model, xgb_model, rfc_model], sample_df, future_period=1, label_type='fwd')
    history_result = eval_models(plain_models, nn_models, test_df, weights=[1, 1], seq_len=10)
    ensemble_pred(plain_models, nn_models, sample_df, seq_len=10, weights=[1, 1], future_period=1, label_type='fwd')
    history_result.to_csv(os.path.join(ROOT_PATH, 'history_result.csv'), index=True, encoding='utf-8')


if __name__ == '__main__':
    main()
