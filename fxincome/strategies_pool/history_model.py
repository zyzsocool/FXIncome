import pandas as pd
import os
from pandas import DataFrame, Series
from scipy.spatial import distance
from fxincome import logger, const


def calculate_all_similarity(df, features: list, metric="cosine"):
    """
    Calculate the Euclidean distance between each pair of rows in a DataFrame for the given features.

    Args:
        df (DataFrame): The input DataFrame containing the data. It must include a 'date' column.
        features (list): The list of feature column names to be used in the distance calculation.
        metric (str): The type of similarity. Must be either "euclidean" or "cosine".
    Returns:
        tuple: A tuple containing two elements:
            - similarity_list (list): A list of tuples where each tuple is of the form (distance, date1, date2).
              The distance is the Euclidean distance between the rows corresponding to date1 and date2.
            - similarity_df (DataFrame): A DataFrame with the same information as in similarity_list,
              but with 'date1' and 'date2' as the index and 'distance' as the column.
    """
    if metric not in ["euclidean", "cosine"]:
        raise ValueError("sim_type must be either 'euclidean' or 'cosine'")

    # Separate the dates from the data
    dates = df["date"]
    data = df[features].values

    # The result is a 2D array where the element at position (i, j) is
    # the distance between the i-th and j-th row of the DataFrame
    distances = distance.cdist(data, data, metric=metric)

    # Create a 2D list of tuples (distance, date1, date2)
    similarity_list = []
    for i in range(len(distances)):
        for j in range(len(distances[i])):
            similarity_list.append((distances[i][j], dates[i], dates[j]))

    # Convert the list of tuples to a DataFrame
    similarity_df = pd.DataFrame(
        similarity_list, columns=["distance", "date_1", "date_2"]
    )
    similarity_df = similarity_df.set_index("distance")

    return similarity_list, similarity_df


def train_test_split(df: DataFrame, train_ratio: float = 0.8, gap: int = 30):
    """
    Split the data into training and testing sets.

    Args:
        df (DataFrame): The input DataFrame containing the data. It must include a 'date' column.
        train_ratio (float): The ratio of the data to be used for training. The rest will be used for testing.
        gap (int): The number of trade days between the end of training set and beginning of testing sets.
    Returns:
        tuple: A tuple containing two DataFrames:
            - train_df: The training set.
            - test_df: The testing set.
    """
    data = df.sort_values(by="date")
    train_size = int(len(data) * train_ratio)
    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size + gap :]
    return train_df, test_df


def avg_yield_chg(
    row,
    simi_matrix_with_yield_chg,
    excluded_dates: Series,
    distance_min: float,
    distance_max: float,
    yield_chg_fwd="yield_chg_fwd_10",
):
    """
    Calculate the weighted average yield change for a given row based on the similarity matrix.

    The weights are calculated as the inverse of the distance, meaning that smaller distances
    will have larger weights and larger distances will have smaller weights. The distances considered
    are those within the range specified by distance_min(included) and distance_max(NOT included).

    Args:
        row (Series): The row for which to calculate the weighted average yield change.
        simi_matrix_with_yield_chg (DataFrame): The similarity matrix with yield changes.
        excluded_dates (Series): The dates to be excluded from the calculation.
        distance_min (float): included.
        distance_max (float): NOT included.
        yield_chg_fwd (str, optional): The column name for the forward yield change. Defaults to "yield_chg_fwd_10".
                                        Its value is associated with simi_matrix_with_yield_chg["date_1"]


    Returns:
        float: The weighted average yield change for the given row. If no similar dates are found, return -99.
    """

    # Find rows with distance in the specified range and the same date as the given row
    query_str = (
        f"`distance` >= {distance_min} "
        f"and `distance` < {distance_max} "
        f"and `date_2` == '{row['date']}' "
        f"and `date_1` not in @excluded_dates"
    )
    close_rows = simi_matrix_with_yield_chg.query(query_str)
    if close_rows.empty:
        return -99
    # Calculate weighted average of yield_chg_fwd, weighted by INVERSE of distance(smaller distance, higher weight)
    weights = 1 / close_rows["distance"]
    weights = weights / weights.sum()
    yield_chg_avg = (close_rows[yield_chg_fwd] * weights).sum()

    # logger.info(row)
    # logger.info(close_rows)
    # logger.info(yield_chg_avg)

    return yield_chg_avg


def predict_yield_chg(
    simi_df: DataFrame,
    test_df: DataFrame,
    distance_min: float,
    distance_max: float,
    yield_chg_fwd="yield_chg_fwd_10",
):
    """
    Predict the yield change direction in n days and calculate the prediction accuracy.

    The yield change direction is determined by yield_chg_fwd in the form of "yield_chg_fwd_n".
    n must be either 5, 10, 20, or 30. The prediction is based on the weighted average yield change
    for each row in the test DataFrame, where the weights are calculated based on the similarity matrix.

    Args:
        simi_df (DataFrame): includes a 'date_1', 'date_2', 'distance', and yield_chg_fwd column.
        test_df (DataFrame): includes a yield_chg_fwd column.
        distance_min (float): included.
        distance_max (float): NOT included
        yield_chg_fwd (str, optional): The column name for the forward yield change. Defaults to "yield_chg_fwd_10".

    Prints:
        The prediction accuracy on test set as a percentage.
    """

    simi_df = simi_df[["date_1", "date_2", "distance", yield_chg_fwd]]

    # If weighted_avg_yield_change() > 0, prediction = 1;
    # if weighted_avg_yield_change() == -99, it cannot find similar dates within distance scope, then prediction = -99;
    # else if weighted_avg_yield_change() <= 0, prediction = 0.
    test_df["prediction"] = test_df.apply(
        lambda row: (
            (lambda x: 1 if x > 0 else -99 if x == -99 else 0)(
                avg_yield_chg(
                    row,
                    simi_df,
                    excluded_dates=test_df["date"],
                    yield_chg_fwd=yield_chg_fwd,
                    distance_min=distance_min,
                    distance_max=distance_max,
                )
            )
        ),
        axis=1,
    )
    test_df["real_chg"] = test_df[yield_chg_fwd].apply(lambda x: 1 if x > 0 else 0)
    # Filter out rows where prediction is -99, which means no similar dates were found.
    valid_predictions = test_df[test_df["prediction"] != -99]
    valid_ratio = len(valid_predictions) / len(test_df)
    accuracy = (valid_predictions["prediction"] == valid_predictions["real_chg"]).mean()
    logger.info(f"valid_prediction size: {len(valid_predictions)}; total size: {len(test_df)}")
    logger.info(f"valid_prediction_ratio: {valid_ratio}")
    logger.info(f"Prediction accuracy: {accuracy * 100}%")


if __name__ == "__main__":
    SIMI_METRIC = "cosine"
    DEST_NAME = f"similarity_matrix_{SIMI_METRIC}.csv"

    data_path = os.path.join(const.PATH.STRATEGY_POOL, "history_processed.csv")
    sample_df = pd.read_csv(data_path)
    sample_df = sample_df.dropna().reset_index(drop=True)

    # _, distance_df = calculate_all_similarity(
    #     sample_df, const.HistorySimilarity.FEATURES, metric=SIMI_METRIC
    # )
    #
    # distance_df.to_csv(
    #     os.path.join(const.PATH.STRATEGY_POOL, DEST_NAME), encoding="utf-8"
    # )

    data_path = os.path.join(const.PATH.STRATEGY_POOL, DEST_NAME)
    distance_df = pd.read_csv(data_path)
    sample_merged = pd.merge(sample_df, distance_df, left_on="date", right_on="date_1")

    train_sample, test_sample = train_test_split(sample_df, train_ratio=0.8, gap=30)

    predict_yield_chg(
        simi_df=sample_merged,
        test_df=test_sample,
        yield_chg_fwd="yield_chg_fwd_10",
        distance_min=0,
        distance_max=0.05
    )
