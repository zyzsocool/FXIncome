import numpy as np
import pandas as pd
import os
import datetime
from fxincome.const import SPREAD
from WindPy import w

w.start()


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
        df.columns = ['CODE', 'IPO_DATE', 'COUPON', 'CLOSE', 'VOL', 'OUT_BAL', 'YTM']
        df.to_csv(output_file, index=True, encoding='utf-8')


def feature_engineering(leg1_code: str, leg2_code: str, days_back: int, n_samples: int):
    """
    Generate X and Y for a pair of bonds. One row of the final dataframe has both features and labels for ONE DAY.
    spread = leg2 - leg1
    leg1's IPO date <= leg2's IPO date
    Samples are selected since leg2's IPO day.
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
    df = pd.merge(df_1, df_2, on='DATE', how='inner', suffixes=(f'_{leg1_code}', f'_{leg2_code}'))
    df.to_csv(SPREAD.SAVE_PATH + f'{leg1_code}_{leg2_code}.csv', index=False, encoding='utf-8')

    # Feature Engineeing
    df['MONTH'] = df['DATE'].dt.month
    # feature: days since ipo of leg2
    df['DAYS_SINCE_LEG2_IPO'] = (df['DATE'] - df['IPO_DATE_' + leg2_code]).dt.days
    df['SPREAD'] = df['YTM_' + leg2_code] - df['YTM_' + leg1_code]
    for i in range(1, days_back + 1):
        df[f'SPREAD_t-{i}'] = df['SPREAD'].shift(i)
    df = df.sort_values(by='DATE')
    first_date = df.iloc[0]['IPO_DATE_' + leg2_code]
    # Yields and other features before leg2's IPO date are doubtful.
    # To calculate features of previous days, we begin from leg2's IPO date + days_back
    # Last sample = leg2's IPO date + days_back + n_samples - 1
    # Only trading days are counted. So it's different from calendar days.
    df = df[(df['DATE'] >= first_date)].iloc[days_back:days_back + n_samples]
    df = df.dropna()
    return df


# download_data()
df = feature_engineering('210210', '210215', 3, 150)
df.to_csv(SPREAD.SAVE_PATH + '210210_210215.csv', index=False, encoding='utf-8')
