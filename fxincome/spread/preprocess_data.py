import numpy as np
import pandas as pd
import datetime
from fxincome.const import PATH, SPREAD
from fxincome import logger
from WindPy import w
from financepy.utils import Date, DayCountTypes, FrequencyTypes, ONE_MILLION
from financepy.products.bonds import Bond, YTMCalcType


def download_data(start_date: datetime.date = datetime.date(2019, 1, 1),
                  end_date: datetime.date = datetime.date(2023, 4, 11)):
    """
    Download data from Wind and save to csv files. One csv for each bond.
    """
    w.start()
    for code in SPREAD.CDB_CODES:
        wind_code = code + '.IB'
        w_data = w.wsd(wind_code,
                       "trade_code,ipo_date, couponrate, close,volume, outstandingbalance, yield_cnbd",
                       start_date,
                       end_date,
                       "credibility=1")
        df = pd.DataFrame(w_data.Data, index=w_data.Fields, columns=w_data.Times)
        df = df.T
        output_file = PATH.SPREAD_DATA + code + '.csv'
        df.index.name = 'DATE'
        df = df.reset_index()
        df.columns = ['DATE', 'CODE', 'IPO_DATE', 'COUPON', 'CLOSE', 'VOL', 'OUT_BAL', 'YTM']
        #  Change NaN to 0 for VOL when IPO_DATE >= DATE. Volume is unlikely to be NaN after IPO.
        df.loc[df.DATE >= df.IPO_DATE, 'VOL'] = df.loc[df.DATE >= df.IPO_DATE, 'VOL'].fillna(0.0)
        df.to_csv(output_file, index=False, encoding='utf-8')


def feature_engineering(leg1_code: str, leg2_code: str, days_back: int, n_samples: int,
                        days_forward: int, spread_threshold: float, features: list[str],
                        keep_date: bool = False) -> pd.DataFrame:
    """
    Generate X and Y for a pair of bonds. One row of the final dataframe has both features and labels for ONE DAY.
    spread = leg2 ytm - leg1 ytm.  YTM's unit is %.
    leg1's IPO date <= leg2's IPO date
    Samples are selected since leg2's IPO day.
    spread_threshold's sign determines whether spread is wider or narrower.
    Labels:
    If spread_threshold is POSITIVE, assuming spread is wider, then:
        during the period between T and T + days_forward,
        if any day's spread - spread_T >= spread_threshold, then label = 1.
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
                        If POSITIVE, assuming spread is wider, then:
                            during the period between T and T + days_forward,
                            if any day's spread - spread_T >= spread_threshold, then label = 1.
                        If NEGATIVE, assuming spread is narrower, then:
                            during the period between T and T + days_forward,
                            if any day's spread - spread_T <= spread_threshold, then label = 1.
        features (list[str]): Only these features are included in the final dataframe.
                              Note that labels are not included in features.
        keep_date (bool): If True, DATE column is included in the final dataframe.
    Returns:
        df(Dataframe): One row of this final dataframe has both features and labels for ONE DAY.
    """
    if leg1_code not in SPREAD.CDB_CODES or leg2_code not in SPREAD.CDB_CODES:
        raise ValueError('Invalid bond code.')
    elif SPREAD.CDB_CODES.index(leg1_code) > SPREAD.CDB_CODES.index(leg2_code):
        raise ValueError('leg1 must be issued before leg2.')
    input_file = PATH.SPREAD_DATA + leg1_code + '.csv'
    df_1 = pd.read_csv(input_file, parse_dates=['DATE', 'IPO_DATE']).drop(columns=['CODE'])
    input_file = PATH.SPREAD_DATA + leg2_code + '.csv'
    df_2 = pd.read_csv(input_file, parse_dates=['DATE', 'IPO_DATE']).drop(columns=['CODE'])
    df = pd.merge(df_1, df_2, on='DATE', how='inner', suffixes=('_LEG1', '_LEG2'))
    df.to_csv(PATH.SPREAD_DATA + f'{leg1_code}_{leg2_code}.csv', index=False, encoding='utf-8')

    # Feature Engineeing
    df['MONTH'] = df.DATE.dt.month
    df['DATE'] = df.DATE.dt.date
    df['IPO_DATE_' + 'LEG2'] = df.IPO_DATE_LEG2.dt.date
    df['DAYS_SINCE_LEG2_IPO'] = (df.DATE - df.IPO_DATE_LEG2).dt.days
    df['SPREAD'] = df.YTM_LEG2 - df.YTM_LEG1
    df['VOL_DIFF'] = df.VOL_LEG2 - df.VOL_LEG1
    df['OUT_BAL_DIFF'] = df.OUT_BAL_LEG2 - df.OUT_BAL_LEG1
    #  MACD of spread
    df['SPREAD_MACD'] = df.SPREAD.ewm(span=days_back / 2, adjust=False).mean() - df.SPREAD.ewm(span=days_back,
                                                                                               adjust=False).mean()
    #  MACD of OUT_BAL_DIFF
    df['OUT_BAL_DIFF_MACD'] = df.OUT_BAL_DIFF.ewm(span=days_back / 2, adjust=False).mean() - df.OUT_BAL_DIFF.ewm(
        span=days_back, adjust=False).mean()
    #  Mean of past [t-1, t-2... t-days_back] days' YTM difference
    df[f'SPREAD_AVG_{days_back}'] = df.SPREAD.rolling(days_back, closed='left').mean()
    # Mean of past [t-1, t-2... t-days_back] days' volume difference
    df[f'VOL_DIFF_AVG_{days_back}'] = df.VOL_DIFF.rolling(days_back, closed='left').mean()
    #  Mean of past [t-1, t-2... t-days_back] days' outstanding balance difference
    df[f'OUT_BAL_DIFF_AVG_{days_back}'] = df.OUT_BAL_DIFF.rolling(days_back, closed='left').mean()
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

    # Label Engineering
    df = df.sort_values(by='DATE')
    if spread_threshold >= 0:  # spread is wider
        df[f'max_in_future'] = df.SPREAD.rolling(
            pd.api.indexers.FixedForwardWindowIndexer(window_size=days_forward + 1)).max()
        df['LABEL'] = df.apply(lambda row: 1 if (row.max_in_future - row.SPREAD) >= spread_threshold else 0,
                               axis=1)
        df = df.drop(columns=['max_in_future'])
    else:  # spread is narrower
        df[f'min_in_future'] = df.SPREAD.rolling(
            pd.api.indexers.FixedForwardWindowIndexer(window_size=days_forward + 1)).min()
        df['LABEL'] = df.apply(lambda row: 1 if (row.min_in_future - row.SPREAD) <= spread_threshold else 0,
                               axis=1)
        df = df.drop(columns=['min_in_future'])

    # Yields and other features before leg2's IPO date are doubtful.
    # To calculate features of previous days, we begin from leg2's IPO date + days_back
    # Last sample = leg2's IPO date + days_back + n_samples - 1
    # Only trading days are counted. It's different from calendar days.
    df = df[(df['DATE'] >= first_date)].iloc[days_back:days_back + n_samples]
    # We suppose our strategy trades only before leg2 - leg1 >= 0 after outstanding balance of leg2 reaches max.
    # Select rows from beginning until spread >=0 after out_bal of leg2 reaches max.
    # If spread never >= 0, then select rows from beginning until out_bal of leg2 reaches max.
    df = df.set_index('DATE')
    max_out_bal = df.OUT_BAL_LEG2.max()
    max_out_bal_df = df[df.OUT_BAL_LEG2 == max_out_bal]
    first_positive_spread_index = max_out_bal_df.SPREAD.ge(0).idxmax()
    df = df.loc[:first_positive_spread_index]
    df = df.reset_index()
    # Ratio: positive samples / total samples
    logger.info(f'Label 1 ratio: {df.LABEL.sum() / len(df):.2f}. Total samples: {len(df)}')
    df = df[['DATE'] + features + ['LABEL']]
    df.to_csv(PATH.SPREAD_DATA + f'{leg1_code}_{leg2_code}_FE.csv', index=False, encoding='utf-8')
    if not keep_date:
        df = df.drop(columns=['DATE'])
    return df


def select_features(days_back: int) -> list[str]:
    """
    Select features you want to keep.
    Modify the needed features in this function.
    Args:
        days_back (int): number of trade days backward for features,
                         when features of these past days are included to this sample.
                         Features are like yields, spreads, volumns, outstanding balances ...
                         for leg1 and leg2 on each past day.
    Returns:
        features (list[str]): list of selected features.
    """

    def dynamic_feature_names(f_name: str, days_back: int = 0, avg: bool = False) -> list[str]:
        """
        If avg is False, return ['f_name', 'f_name_t-1', 'f_name_t-2', ..., 'f_name_t-{days_back}']
        If avg is True, return ['f_name', 'f_name_AVG_{days_back}']
        """
        feature_names = [f_name]
        if not avg:
            for i in range(1, days_back + 1):
                feature_names.append(f'{f_name}_t-{i}')
        else:
            feature_names.append(f'{f_name}_AVG_{days_back}')
        return feature_names

    # Features for XGB
    # features = ['YTM_LEG1', 'YTM_LEG2', 'DAYS_SINCE_LEG2_IPO']
    # features += dynamic_feature_names('SPREAD', days_back=days_back, avg=True)
    # features += dynamic_feature_names('VOL_DIFF', days_back=days_back, avg=True)
    # features += dynamic_feature_names('OUT_BAL_DIFF', days_back=days_back, avg=True)

    # Features for LR
    # features = []
    # features += dynamic_feature_names('SPREAD', days_back=days_back, avg=True)
    # features += dynamic_feature_names('VOL_DIFF', days_back=days_back, avg=True)
    # features += dynamic_feature_names('OUT_BAL_DIFF', days_back=days_back, avg=True)

    # Features for LGBM
    features = []  # 'MONTH''YTM_LEG1', 'YTM_LEG2', 'DAYS_SINCE_LEG2_IPO'
    features += dynamic_feature_names('VOL_LEG1', days_back=days_back)
    features += dynamic_feature_names('VOL_LEG2', days_back=days_back)
    features += dynamic_feature_names('YTM_LEG1', days_back=days_back)
    features += dynamic_feature_names('YTM_LEG2', days_back=days_back)
    features += dynamic_feature_names('SPREAD', days_back=days_back)
    features += dynamic_feature_names('VOL_DIFF', days_back=days_back)
    features += dynamic_feature_names('OUT_BAL_DIFF', days_back=1)
    return features


def prepare_backtest_data(leg1_code: str, leg2_code: str):
    """
    Prepare data for backtest and save to csv file. One csv for each pair of bonds.
    The full prices are calculated from YTMs.
    """
    if leg1_code not in SPREAD.CDB_CODES or leg2_code not in SPREAD.CDB_CODES:
        raise ValueError('Invalid bond code.')
    elif SPREAD.CDB_CODES.index(leg1_code) > SPREAD.CDB_CODES.index(leg2_code):
        raise ValueError('leg1 must be issued before leg2.')
    w.start()
    input_file = PATH.SPREAD_DATA + leg1_code + '.csv'
    df_1 = pd.read_csv(input_file, parse_dates=['DATE', 'IPO_DATE'])
    input_file = PATH.SPREAD_DATA + leg2_code + '.csv'
    df_2 = pd.read_csv(input_file, parse_dates=['DATE', 'IPO_DATE'])
    input_file = PATH.SPREAD_DATA + 'bond_lending_rate.csv'
    df_lend = pd.read_csv(input_file, parse_dates=['DATE'])
    df_lend = df_lend[~df_lend['CODE'].str.contains('QF')]
    df_lend['CODE'] = df_lend['CODE'].astype('int64')
    df1 = pd.merge(df_1, df_lend, on=['DATE', 'CODE'], how='left')
    df1.loc[:, ['LEND_RATE']] = df1.loc[:, ['LEND_RATE']].fillna(method='ffill')
    df1.loc[:, ['LEND_RATE']] = df1.loc[:, ['LEND_RATE']].fillna(method='bfill')
    df2 = pd.merge(df_2, df_lend, on=['DATE', 'CODE'], how='left')
    df2.loc[:, ['LEND_RATE']] = df2.loc[:, ['LEND_RATE']].fillna(method='ffill')
    df2.loc[:, ['LEND_RATE']] = df2.loc[:, ['LEND_RATE']].fillna(method='bfill')

    def get_bond(leg_code: str):
        error_code, rows = w.wss(leg_code + '.IB', 'carrydate,maturitydate,couponrate,interestfrequency', usedf=True)
        row = rows.iloc[0]
        issue_date = Date(row['CARRYDATE'].day, row['CARRYDATE'].month, row['CARRYDATE'].year)
        maturity_date = Date(row['MATURITYDATE'].day, row['MATURITYDATE'].month, row['MATURITYDATE'].year)
        coupon = row['COUPONRATE'] / 100
        accrual_type = DayCountTypes.ACT_ACT_ICMA
        face = ONE_MILLION * 500
        if row['INTERESTFREQUENCY'] == 1:
            freq_type = FrequencyTypes.ANNUAL
        elif row['INTERESTFREQUENCY'] == 2:
            freq_type = FrequencyTypes.SEMI_ANNUAL
        elif row['INTERESTFREQUENCY'] == 4:
            freq_type = FrequencyTypes.QUARTERLY
        else:
            raise ValueError(f'Unknown frequency type {row["INTERESTFREQUENCY"]}')
        bond = Bond(issue_date, maturity_date, coupon, freq_type, accrual_type, face)
        return bond

    bond1 = get_bond(leg1_code)
    bond2 = get_bond(leg2_code)
    df1 = df1[['DATE'] + ['CODE'] + ['VOL'] + ['OUT_BAL'] + ['YTM'] + ['LEND_RATE']]
    df2 = df2[['DATE'] + ['CODE'] + ['VOL'] + ['OUT_BAL'] + ['YTM'] + ['LEND_RATE']]
    df = pd.merge(df1, df2, on='DATE', how='inner', suffixes=('_LEG1', '_LEG2'))
    df['SPREAD'] = df['YTM_LEG2'] - df['YTM_LEG1']
    df = df.dropna()
    df = df.set_index('DATE')
    df = df.reset_index()
    df['ESTIMATED_PRICE1'] = np.nan
    df['ESTIMATED_PRICE2'] = np.nan
    for i in range(0, len(df)):
        settlement_date = Date(df['DATE'][i].day, df['DATE'][i].month, df['DATE'][i].year)
        df.loc[i, 'ESTIMATED_PRICE1'] = bond1.full_price_from_ytm(settlement_date, df.loc[i, 'YTM_LEG1'] / 100,
                                                                  YTMCalcType.US_STREET)
        df.loc[i, 'ESTIMATED_PRICE2'] = bond2.full_price_from_ytm(settlement_date, df.loc[i, 'YTM_LEG2'] / 100,
                                                                  YTMCalcType.US_STREET)
    df['Close'] = df['ESTIMATED_PRICE1'] - df['ESTIMATED_PRICE2']
    df['Open'] = df['ESTIMATED_PRICE1'] - df['ESTIMATED_PRICE2']
    df['High'] = df['ESTIMATED_PRICE1'] - df['ESTIMATED_PRICE2']
    df['Low'] = df['ESTIMATED_PRICE1'] - df['ESTIMATED_PRICE2']
    max_out_bal = df.OUT_BAL_LEG2.max()
    max_out_bal_df = df[df.OUT_BAL_LEG2 == max_out_bal]
    # Select rows from beginning until spread >=0 after out_bal of leg2 reaches max.
    # If spread never >= 0, select all data.
    if sum(max_out_bal_df.SPREAD.ge(0)) > 0:
        first_positive_spread_index = max_out_bal_df.SPREAD.ge(0).idxmax()
        df = df.loc[:first_positive_spread_index + 1]
    df = df.reset_index()
    df = df.drop(['index', 'ESTIMATED_PRICE1', 'ESTIMATED_PRICE2'], axis=1)
    df.to_csv(PATH.SPREAD_DATA + leg1_code + '_' + leg2_code + '_bt.csv', index=False, encoding='utf-8')


# download_data(start_date=datetime.date(2019, 1, 10), end_date=datetime.date(2023, 5, 4))
for i in range(0, 13):
    prepare_backtest_data(SPREAD.CDB_CODES[i], SPREAD.CDB_CODES[i + 1])
# features = select_features(days_back=4)
# feature_engineering('220205', '220210', days_back=4, n_samples=200, days_forward=10, spread_threshold=0.01,
#                     features=features)
