import pandas as pd
import os
from pandas import DataFrame
from scipy.spatial import distance
from fxincome import logger, const


def calculate_all_similarity(df, features: list):
    """
    Calculate the Euclidean distance between each pair of rows in a DataFrame for the given features.

    Args:
        df (DataFrame): The input DataFrame containing the data. It must include a 'date' column.
        features (list): The list of feature column names to be used in the distance calculation.

    Returns:
        tuple: A tuple containing two elements:
            - similarity_list (list): A list of tuples where each tuple is of the form (distance, date1, date2).
              The distance is the Euclidean distance between the rows corresponding to date1 and date2.
            - similarity_df (DataFrame): A DataFrame with the same information as in similarity_list,
              but with 'date1' and 'date2' as the index and 'distance' as the column.
    """

    # Separate the dates from the data
    dates = df["date"]
    data = df[features].values

    # The result is a 2D array where the element at position (i, j) is
    # the distance between the i-th and j-th row of the DataFrame
    distances = distance.cdist(data, data, "euclidean")

    # Create a 2D list of tuples (distance, date1, date2)
    similarity_list = []
    for i in range(len(distances)):
        for j in range(len(distances[i])):
            similarity_list.append((distances[i][j], dates[i], dates[j]))

    # Convert the list of tuples to a DataFrame
    similarity_df = pd.DataFrame(
        similarity_list, columns=["distance", "date_1", "date_2"]
    )
    similarity_df = similarity_df.set_index(["date_1", "date_2"])

    return similarity_list, similarity_df


if __name__ == "__main__":
    DEST_NAME = "similarity_matrix.csv"
    data_path = os.path.join(const.PATH.STRATEGY_POOL, "history_processed.csv")
    df = pd.read_csv(data_path)
    df = df.dropna().reset_index(drop=True)

    _, distance_df = calculate_all_similarity(df, const.HistorySimilarity.FEATURES)

    distance_df.to_csv(os.path.join(const.PATH.STRATEGY_POOL, DEST_NAME), encoding="utf-8")