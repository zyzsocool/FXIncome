import pandas as pd
import os
from pandas import DataFrame
from fxincome import logger, const

data_path = os.path.join(const.PATH.STRATEGY_POOL, "history_processed.csv")
all_samples = pd.read_csv(data_path)
all_samples = all_samples.dropna().reset_index(drop=True)

for day, name in const.HistorySimilarity.LABELS.items():
    all_samples[f"actual_{day}"] = all_samples[name].apply(lambda x: 1 if x > 0 else 0)

# Print the ratios of positive values in all_samples[f"actual_{day}"]
for day in const.HistorySimilarity.LABELS.keys():
    logger.info(f"Actual positive ratio for {day}: {all_samples[f'actual_{day}'].sum()/len(all_samples)}")