import numpy as np
import pandas as pd
import os

from ydata_profiling import ProfileReport

ROOT_PATH = "d:/ProjectRicequant/fxincome/strategies_pool"
# SRC_NAME = "history_processed.csv"
SRC_NAME = "history_similarity.csv"

data = pd.read_csv(os.path.join(ROOT_PATH, SRC_NAME), parse_dates=["date"])
profile = ProfileReport(data, title="Pandas Profiling Report")
profile.to_file("d:/history_similarity_stats.html")
