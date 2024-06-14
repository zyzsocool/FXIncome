import numpy as np
import pandas as pd
import os

from ydata_profiling import ProfileReport

ROOT_PATH = "d:/ProjectRicequant/fxincome/strategies_pool"
SRC_NAME = "similarity_matrix_cosine.csv"
# SRC_NAME = "similarity_matrix_euclidean.csv"

data = pd.read_csv(os.path.join(ROOT_PATH, SRC_NAME), parse_dates=["date_1","date_2"])
profile = ProfileReport(data, title="Pandas Profiling Report")
profile.to_file(f"d:/{SRC_NAME}.html")
