import pandas as pd
from functools import reduce
# First DataFrame
data1 = {
    'date': ['2023/1/2', '2023/2/1'],
    '2024/7/5': [0.5, 0.7],
    'yield_chg_fwd_5': [0.02, 0.03]
}
df1 = pd.DataFrame(data1)

# Second DataFrame
data2 = {
    'date': ['2023/1/2', '2023/2/1'],
    '2024/7/5': [0.5, 0.7],
    'yield_chg_fwd_10': [0.1, 0.5]
}
df2 = pd.DataFrame(data2)

# Third DataFrame
data3 = {
    'date': ['2023/1/2', '2023/2/1'],
    '2024/7/5': [0.5, 0.7],
    'yield_chg_fwd_20': [0.1, 0.5]
}
df3 = pd.DataFrame(data3)

# Assuming df1, df2, df3 are your DataFrames
dataframes = [df1, df2, df3]

combined_df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dataframes)

print(combined_df)