import pandas as pd
import os

ROOT_PATH = "d:/ProjectRicequant/fxincome/strategies_pool"
SRC_NAME = "history_processed.csv"

data = pd.read_csv(os.path.join(ROOT_PATH, SRC_NAME), parse_dates=["date"])
# Assuming 'df' is your DataFrame and 'date' is the column with the dates
trading_days_per_year = data.resample('Y', on='date').size()
print(trading_days_per_year)