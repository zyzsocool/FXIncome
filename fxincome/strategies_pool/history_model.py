import pandas as pd
import os
from pandas import DataFrame, Series
from fxincome import logger, const


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
    test_df = data.iloc[(train_size + gap) :]

    logger.info(f"train size: {len(train_df)}; test size: {len(test_df)}")
    logger.info(
        f"train start date: {train_df['date'].iloc[0]}; train end date: {train_df['date'].iloc[-1]}"
    )
    logger.info(
        f"test start date: {test_df['date'].iloc[0]}; test end date: {test_df['date'].iloc[-1]}"
    )
    return train_df, test_df


def avg_yield_chg(
    row,
    simi_matrix_with_yield_chg,
    distance_min: float,
    distance_max: float,
    yield_chg_fwd="yield_chg_fwd_10",
):
    """
    Calculate the weighted average yield change for a given date based on the similarity matrix.

    The weights are calculated as the inverse of the distance, meaning that smaller distances
    will have larger weights and larger distances will have smaller weights. The distances considered
    are those within the range specified by distance_min(included) and distance_max(NOT included).

    Args:
        row (Series): The row for which to calculate the weighted average yield change. row["date"] is the given date.
        simi_matrix_with_yield_chg (DataFrame): The similarity matrix with yield changes.
        distance_min (float): included.
        distance_max (float): NOT included.
        yield_chg_fwd (str, optional): The column name for the forward yield change. Defaults to "yield_chg_fwd_10".

    Returns:
        float: The weighted average yield change for the given row. If no similar dates are found, return -99.
    """

    given_date = row["date"]
    # Find the column with the same date as the given date
    close_rows = simi_matrix_with_yield_chg[["date", given_date, yield_chg_fwd]]
    # Find rows with distance in the specified range
    close_rows = close_rows[
        (close_rows[given_date] >= distance_min)
        & (close_rows[given_date] < distance_max)
    ]

    if close_rows.empty:
        return -99
    # Calculate weighted average of yield_chg_fwd, weighted by INVERSE of distance(smaller distance, higher weight)
    c = 0.5  # 'c' is used to smooth the weight differences caused by the inverse of the distance.
    weights = 1 / (close_rows[given_date] + c)
    weights = weights / weights.sum()
    yield_chg_avg = (close_rows[yield_chg_fwd] * weights).sum()

    return yield_chg_avg


def predict_yield_chg(
    simi_df: DataFrame,
    sample_df: DataFrame,
    yield_chg_fwd: str,
    distance_min: float,
    distance_max: float,
    train_ratio: float,
    gap: int,
):
    """
    Predict the yield change direction in n days and calculate the prediction accuracy.

    The yield change direction is determined by yield_chg_fwd in the form of "yield_chg_fwd_n".
    n must be either 5, 10, 20, or 30. The prediction is based on the weighted average yield change
    for each row in the test DataFrame, where the weights are calculated based on the similarity matrix.

    Args:
        simi_df (DataFrame): includes column 'date'(m rows), column ['2010-1-27', '2011-12-1'...](also m columns)
        sample_df (DataFrame): includes column 'date' and column 'yield_chg_fwd'.
        yield_chg_fwd (str, optional): The column name for the forward yield change. Defaults to "yield_chg_fwd_10".
        distance_min (float): included.
        distance_max (float): NOT included
        train_ratio (float): The ratio of the data to be used for training. The rest will be used for testing.
        gap (int): The number of trade days between the end of training set and beginning of testing sets.
    Prints:
        The prediction accuracy on test set as a percentage.
    """

    # Add column 'yield_chg_fwd' to simi_df.
    # simi_df includes column 'date'(m rows), column ['2010-1-27', '2011-12-1'...](also m columns)
    # and column 'yield_chg_fwd'
    simi_df = pd.merge(sample_df, simi_df, left_on="date", right_on="date")
    train_df, test_df = train_test_split(sample_df, train_ratio=train_ratio, gap=gap)

    # Dates in history are used to calculate weighted average yield change.
    # Rows on test_df["date"] should NOT be included, since they are not in history.
    simi_df = simi_df[~simi_df["date"].isin(test_df["date"])]

    # If avg_yield_chg() > 0, prediction = 1;
    # if avg_yield_chg() == -99, it cannot find similar dates within distance scope, then prediction = -99;
    # else if avg_yield_chg() <= 0, prediction = 0.
    test_df["prediction"] = test_df.apply(
        lambda row: (
            (lambda x: 1 if x > 0 else -99 if x == -99 else 0)(
                avg_yield_chg(
                    row,
                    simi_df,
                    yield_chg_fwd=yield_chg_fwd,
                    distance_min=distance_min,
                    distance_max=distance_max,
                )
            )
        ),
        axis=1,
    )
    test_df["real_chg"] = test_df[yield_chg_fwd].apply(lambda x: 1 if x > 0 else 0)
    # Filter out rows where predictions are -99, which means no similar dates were found.
    valid_predictions = test_df[test_df["prediction"] != -99]
    valid_ratio = len(valid_predictions) / len(test_df)
    accuracy = (valid_predictions["prediction"] == valid_predictions["real_chg"]).mean()
    logger.info(
        f"valid_prediction size: {len(valid_predictions)}; total size: {len(test_df)}"
    )
    logger.info(f"valid_prediction_ratio: {valid_ratio}")
    logger.info(f"Prediction accuracy: {accuracy * 100}%")


if __name__ == "__main__":
    MATRIX_EUCLIDEAN = f"similarity_matrix_euclidean.csv"
    MATRIX_COSINE = f"similarity_matrix_cosine.csv"
    YIELD_CHG_FWD_COL = "yield_chg_fwd_10"

    data_path = os.path.join(const.PATH.STRATEGY_POOL, "history_processed.csv")
    all_samples = pd.read_csv(data_path)
    all_samples = all_samples.dropna().reset_index(drop=True)
    all_samples = all_samples[["date", YIELD_CHG_FWD_COL]]

    data_path = os.path.join(const.PATH.STRATEGY_POOL, MATRIX_COSINE)
    distance_df = pd.read_csv(data_path)

    predict_yield_chg(
        simi_df=distance_df,
        sample_df=all_samples,
        yield_chg_fwd=YIELD_CHG_FWD_COL,
        distance_min=0,
        distance_max=0.24,
        train_ratio=0.85,
        gap=30,
    )
