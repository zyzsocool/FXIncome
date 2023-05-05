import pandas as pd
import numpy as np
import datetime
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report
from fxincome.spread.preprocess_data import feature_engineering
from fxincome.utils import ModelAttr, JsonModel
from fxincome.const import PATH, SPREAD
from fxincome import logger


def predict_pair_spread(model_name: str, leg1_code: str, leg2_code: str) -> pd.DataFrame:
    """
    Load a scikit-learn like classifier by name. Predict the spread of a pair of bonds using downloaded data from Wind.
    Spread = leg2 YTM - leg1 YTM
    The model attributes are loaded from a json file.
    Args:
        model_name: include the file extension, e.g. 'model.pkl', 'model.json'
        leg1_code: the wind code of the first leg.
        leg2_code: the wind code of the second leg.
    Returns:
        Dataframe with predictions.
    """
    model_attr = JsonModel.load_attr(model_name, PATH.SPREAD_MODEL + JsonModel.JSON_NAME)
    if model_attr is None:
        raise ValueError(f'No model named {model_name} found in {PATH.SPREAD_MODEL + JsonModel.JSON_NAME}')
    df = feature_engineering(
        leg1_code=leg1_code,
        leg2_code=leg2_code,
        days_back=model_attr.other['days_back'],
        n_samples=model_attr.other['n_samples'],
        days_forward=model_attr.labels['LABEL']['days_forward'],
        spread_threshold=model_attr.labels['LABEL']['spread_threshold'],
        features=model_attr.features,
        keep_date=True
    )
    #  Load model
    if ('LR' in model_name) or ('LGB' in model_name):
        model = joblib.load(PATH.SPREAD_MODEL + model_name + '.pkl')
    elif 'XGB' in model_name:
        booster = xgb.Booster()
        booster.load_model(PATH.SPREAD_MODEL + model_name + '.json')
        model = xgb.XGBClassifier()
        model._Booster = booster
    else:
        raise NotImplementedError(f"file type {model_name.split('.')[-1]} is not supported")
    #  Predict
    X = df[model_attr.features]
    Y = df[model_attr.labels.keys()]
    preds = model.predict(X)
    score = model.score(X, Y)
    logger.info(f'Mean Accuracy of Model {model_name}: {score}')
    print(classification_report(Y, preds))
    probs = model.predict_proba(X)
    df.insert(len(df.columns), column='pred', value=preds)
    df.insert(len(df.columns), column='actual', value=Y)
    df['correctness'] = df.apply(lambda x: 1 if x.pred == x.actual else 0, axis=1)
    df['prob_0'] = probs[:, 0]
    df['prob_1'] = probs[:, 1]
    history_result = df[['DATE', 'correctness', 'pred', 'actual', 'prob_0', 'prob_1']].copy()
    spread_threshold = model_attr.labels['LABEL']['spread_threshold']
    history_result.to_csv(
        PATH.SPREAD_DATA + f'predictions_{model_name}_{leg1_code}_{leg2_code}_{spread_threshold}.csv',
        index=False)
    return history_result


def main():
    predict_pair_spread('spread_0.635_XGB_20230428_1029', '220220', '230205')


if __name__ == '__main__':
    main()
