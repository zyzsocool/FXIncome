import scipy.stats as stats

# Define the set
data_set = [1, 2, 3, 4,5]

# Calculate the percentile rank
percentile_rank = stats.percentileofscore(data_set, 3) / 100

print(percentile_rank)