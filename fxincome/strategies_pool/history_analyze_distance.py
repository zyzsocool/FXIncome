import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from fxincome import const, logger


def distance_histogram(distance_df):
    # Convert 'date_1' and 'date_2' to datetime format
    distance_df["date_1"] = pd.to_datetime(distance_df["date_1"])
    distance_df["date_2"] = pd.to_datetime(distance_df["date_2"])
    # Calculate the absolute difference in days between 'date_1' and 'date_2'
    distance_df["date_diff"] = (
        (distance_df["date_1"] - distance_df["date_2"]).abs().dt.days
    )
    # Define the date ranges
    max_diff = distance_df["date_diff"].max()
    n_ranges = 100
    logger.info(f"max_date_diff: {max_diff}")
    logger.info(f"n_ranges: {n_ranges}")
    ranges = [
        (i * max_diff / n_ranges, (i + 1) * max_diff / n_ranges)
        for i in range(n_ranges)
    ]
    # Calculate the average distance for each range
    avg_distances = []
    for date_range in ranges:
        avg_distance = distance_df[
            (distance_df["date_diff"] > date_range[0])
            & (distance_df["date_diff"] <= date_range[1])
        ]["distance"].mean()
        avg_distances.append(avg_distance)
    # Create a DataFrame with the average distances and their corresponding ranges
    avg_distances_df = pd.DataFrame(
        {
            "Range": [f"{int(r[0])}-{int(r[1])}" for r in ranges],
            "Average Distance": avg_distances,
        }
    )
    # Plot the average distances for each range
    plt.figure(figsize=(10, 6))
    plt.bar(avg_distances_df["Range"], avg_distances_df["Average Distance"])
    plt.xlabel("Date Range (days)")
    plt.ylabel("Average Distance")
    plt.title("Average Distance for Different Date Ranges")
    plt.xticks(rotation=45)
    plt.show()


def check_date(distance_df):
    # Check if each date in 'date_1' is also present in 'date_2'
    date_check_1 = distance_df["date_1"].isin(distance_df["date_2"])
    date_check_2 = distance_df["date_2"].isin(distance_df["date_1"])
    # Print the result
    print(f"All date_1 are in date_2: {date_check_1.all()}")
    print(f"All date_2 are in date_1: {date_check_2.all()}")


SIM_METRIC = "euclidean"
SRC_NAME = f"similarity_matrix_{SIM_METRIC}.csv"

data_path = os.path.join(const.PATH.STRATEGY_POOL, SRC_NAME)
df = pd.read_csv(data_path)
# distance_histogram(df)
check_date(df)
