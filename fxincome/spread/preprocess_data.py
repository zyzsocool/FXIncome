import numpy as np
import pandas as pd
import os
import datetime
from fxincome.const import SPREAD
from WindPy import w
from fxincome import logger

w.start()


def dynamic_feature_names(f_name: str, days_back: int = 0):
    feature_names = [f_name]
    for i in range(1, days_back + 1):
        feature_names.append(f'{f_name}_t-{i}')
    return feature_names


def download_data(start_date: datetime.date = datetime.date(2019, 1, 1),
                  end_date: datetime.date = datetime.date(2023, 3, 3)):
    for code in SPREAD.CDB_CODES:
        wind_code = code + '.IB'
        w_data = w.wsd(wind_code,
                       "trade_code,ipo_date, couponrate, close,volume, outstandingbalance, yield_cnbd",
                       start_date,
                       end_date,
                       "credibility=1")
        df = pd.DataFrame(w_data.Data, index=w_data.Fields, columns=w_data.Times)
        df = df.T
        output_file = SPREAD.SAVE_PATH + code + '.csv'
        df.index.name = 'DATE'
        df = df.reset_index()
        df.columns = ['DATE', 'CODE', 'IPO_DATE', 'COUPON', 'CLOSE', 'VOL', 'OUT_BAL', 'YTM']
        #  Change NaN to 0 for VOL when IPO_DATE >= DATE. Volume is unlikely to be NaN after IPO.
        df.loc[df.DATE >= df.IPO_DATE, 'VOL'] = df.loc[df.DATE >= df.IPO_DATE, 'VOL'].fillna(0)
        df.to_csv(output_file, index=False, encoding='utf-8')


def feature_engineering(leg1_code: str, leg2_code: str, days_back: int, n_samples: int,
                        days_forward: int, spread_threshold: float) -> pd.DataFrame:
    """
    Generate X and Y for a pair of bonds. One row of the final dataframe has both features and labels for ONE DAY.
    spread = leg2 ytm - leg1 ytm.  YTM's unit is %.
    leg1's IPO date <= leg2's IPO date
    Samples are selected since leg2's IPO day.
    spread_threshold's sign determines whether spread is wider or narrower.
    Labels:
    If spread_threshold is POSITIVE, assuming spread is wider, then:
        during the period between T and T + days_forward,
        if any day's spread - spread_T > spread_threshold, then label = 1.
    If spread_threshold is NEGATIVE, assuming spread is narrower, then:
        during the period between T and T + days_forward,
        if any day's spread - spread_T < spread_threshold, then label = 1.
    Args:
        leg1_code (str): leg1 bond code.
        leg2_code (str): leg2 bond code.
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
    Returns:
        df(Dataframe): One row of this final dataframe has both features and labels for ONE DAY.
    """
    if leg1_code not in SPREAD.CDB_CODES or leg2_code not in SPREAD.CDB_CODES:
        raise ValueError('Invalid bond code.')
    elif SPREAD.CDB_CODES.index(leg1_code) > SPREAD.CDB_CODES.index(leg2_code):
        raise ValueError('leg1 must be issued before leg2.')
    input_file = SPREAD.SAVE_PATH + leg1_code + '.csv'
    df_1 = pd.read_csv(input_file, parse_dates=['DATE', 'IPO_DATE']).drop(columns=['CODE'])
    input_file = SPREAD.SAVE_PATH + leg2_code + '.csv'
    df_2 = pd.read_csv(input_file, parse_dates=['DATE', 'IPO_DATE']).drop(columns=['CODE'])
    df = pd.merge(df_1, df_2, on='DATE', how='inner', suffixes=('_LEG1', '_LEG2'))
    df.to_csv(SPREAD.SAVE_PATH + f'{leg1_code}_{leg2_code}.csv', index=False, encoding='utf-8')

    # Feature Engineeing
    df['MONTH'] = df.DATE.dt.month
    df['DATE'] = df.DATE.dt.date
    df['IPO_DATE_' + 'LEG2'] = df.IPO_DATE_LEG2.dt.date
    df['DAYS_SINCE_LEG2_IPO'] = (df.DATE - df.IPO_DATE_LEG2).dt.days
    df['SPREAD'] = df.YTM_LEG2 - df.YTM_LEG1
    df['VOL_DIFF'] = df.VOL_LEG2 - df.VOL_LEG1
    df['OUT_BAL_DIFF'] = df.OUT_BAL_LEG2 - df.OUT_BAL_LEG1
    for i in range(1, days_back + 1):
        df[f'SPREAD_t-{i}'] = df.SPREAD.shift(i)
    for i in range(1, days_back + 1):
        df[f'VOL_DIFF_t-{i}'] = df.VOL_DIFF.shift(i)
    for i in range(1, days_back + 1):
        df[f'OUT_BAL_DIFF_t-{i}'] = df.OUT_BAL_DIFF.shift(i)
    for i in range(1, days_back + 1):
        df[f'VOL_LEG1_t-{i}'] = df.VOL_LEG1.shift(i)
    for i in range(1, days_back + 1):
        df[f'VOL_LEG2_t-{i}'] = df.VOL_LEG2.shift(i)
    for i in range(1, days_back + 1):
        df[f'YTM_LEG1_t-{i}'] = df.YTM_LEG1.shift(i)
    for i in range(1, days_back + 1):
        df[f'YTM_LEG2_t-{i}'] = df.YTM_LEG2.shift(i)
    for i in range(1, days_back + 1):
        df[f'OUT_BAL_LEG1_t-{i}'] = df.OUT_BAL_LEG1.shift(i)
    for i in range(1, days_back + 1):
        df[f'OUT_BAL_LEG2_t-{i}'] = df.OUT_BAL_LEG2.shift(i)

    first_date = df.iloc[0]['IPO_DATE_LEG2']  # leg2's IPO date, for selecting n samples.
    df = select_features(df, days_back=days_back)

    # Label Engineering
    df = df.sort_values(by='DATE')
    if spread_threshold >= 0:  # spread is wider
        df[f'max_in_future'] = df.SPREAD.rolling(
            pd.api.indexers.FixedForwardWindowIndexer(window_size=days_forward + 1)).max()
        df['LABEL'] = df.apply(lambda row: 1 if (row.max_in_future - row.SPREAD) > spread_threshold else 0,
                               axis=1)
        df = df.drop(columns=['max_in_future'])
    else:  # spread is narrower
        df[f'min_in_future'] = df.SPREAD.rolling(
            pd.api.indexers.FixedForwardWindowIndexer(window_size=days_forward + 1)).min()
        df['LABEL'] = df.apply(lambda row: 1 if (row.min_in_future - row.SPREAD) < spread_threshold else 0,
                               axis=1)
        df = df.drop(columns=['min_in_future'])

    # Yields and other features before leg2's IPO date are doubtful.
    # To calculate features of previous days, we begin from leg2's IPO date + days_back
    # Last sample = leg2's IPO date + days_back + n_samples - 1
    # Only trading days are counted. It's different from calendar days.
    df = df[(df['DATE'] >= first_date)].iloc[days_back:days_back + n_samples]
    df.to_csv(SPREAD.SAVE_PATH + f'{leg1_code}_{leg2_code}_FE.csv', index=False, encoding='utf-8')
    # Ratio: positive samples / total samples
    logger.info(f'Label 1 ratio: {df.LABEL.sum() / len(df):.2f}')
    return df


def select_features(df: pd.DataFrame, days_back: int) -> pd.DataFrame:
    """
    Select features and return the selected dataframe.
    Modify the needed features in this function.
    Args:
        df (Dataframe): Dataframe to select columns from.
        days_back (int): number of trade days backward for features,
                         when features of these past days are included to this sample.
                         Features are like yields, spreads, volumns, outstanding balances ...
                         for leg1 and leg2 on each past day.
    Returns:
        df (Dataframe): One row of this final dataframe has both features and labels for ONE DAY.
    """
    features = ['DATE', 'MONTH', 'DAYS_SINCE_LEG2_IPO', 'YTM_LEG1', 'YTM_LEG2']
    features += dynamic_feature_names('SPREAD', days_back=days_back)
    features += dynamic_feature_names('VOL_DIFF', days_back=days_back)
    features += dynamic_feature_names('OUT_BAL_DIFF', days_back=days_back)
    df = df[features]
    return df


# download_data()
feature_engineering('210215', '220205', days_back=5, n_samples=150, days_forward=10, spread_threshold=0.01)
