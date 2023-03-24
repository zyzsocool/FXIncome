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
        w_data = w.wsd(code,
                       "trade_code,ipo_date, couponrate, close,volume, outstandingbalance, yield_cnbd",
                       start_date,
                       end_date,
                       "credibility=1")
        df = pd.DataFrame(w_data.Data, index=w_data.Fields, columns=w_data.Times)
        df = df.T
        output_file = SPREAD.SAVE_PATH + code + '.csv'
        df.index.name = 'date'
        df.to_csv(output_file, index=True, encoding='utf-8')


def read_data_from_disk(start_date: datetime.date = datetime.date(2019, 1, 1),
                        end_date: datetime.date = datetime.date(2023, 3, 3)):
    all_data = []
    ipo_dates = []
    for file_name in os.listdir(SPREAD.SAVE_PATH):
        file = os.path.join(SPREAD.SAVE_PATH, file_name)
        df = pd.read_csv(file, parse_dates=['date', 'IPO_DATE'])
        df.date = df.date.dt.date
        df = df[(df.date >= start_date) & (df.date <= end_date)]
        ipo_dates.append(df.IPO_DATE[0].to_pydatetime().date())
        all_data.append(df)
    return all_data, ipo_dates


def generate_x(leg1_code: str, leg2_code: str, all_data: list, ipo_dates: list,
               feature_window: int, sample_window: int):
    '''
    Generate X for a pair of bonds. X is a 2D array with shape (days_since_ipo, n_features).
    spread = leg1 - leg2
    leg1's IPO date <= leg2's IPO date
    Samples are selected from data since leg2's IPO day.
    Args:
        leg1_code(str): leg1 bond code.
        leg2_code(str): leg2 bond code.
        all_data(list): all data read from disk.
        feature_window(int): number of days backward for features,
                            when features of these past days are included to this sample.
                            Features are like yields, spreads, volumns, outstanding balances ...
                            for leg1 and leg2 on each past day.
        sample_window(int): number of samples selected from data. Samples are selected from data since leg2's IPO day.
                            last day = leg2's IPO day + sample_window - 1
        ipo_dates: list of ipo dates of SPREAD.CDB_CODES.
    Returns:
        x(list): X is a 2D array with shape (days_since_ipo, n_features).
    '''
    timeline = all_data[0].date.values  # days of samples
    n_samples = len(timeline)
    leg1_index = SPREAD.CDB_CODES.index(leg1_code)
    leg2_index = SPREAD.CDB_CODES.index(leg2_code)
    spread = all_data[leg1_index].YIELD_CNBD - all_data[leg2_index].YIELD_CNBD
    # feature: exact month of each sample
    months = [timeline[i].month for i in range(n_samples)]
    # feature: days since ipo of leg2
    days_from_leg2_ipo = [(timeline[i] - ipo_dates[leg2_index]).days for i in range(n_samples)]

# download_data()
all_data, ipo_dates = read_data_from_disk()
generate_x('190210.IB', '190215.IB', all_data, ipo_dates, 5, 5)
