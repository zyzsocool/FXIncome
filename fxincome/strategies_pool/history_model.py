import numpy as np
import pandas as pd
import os, logging
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from pandas import DataFrame
from fxincome import logger, handler, const
from fxincome.strategies_pool.history_process_data import (
    feature_engineering,
    calculate_all_similarity,
)


def print_prediction_stats(y_actual, y_pred):
    logger.info(f"Accuracy: {accuracy_score(y_actual, y_pred) * 100:.2f}%")
    logger.info(confusion_matrix(y_actual, y_pred))
    logger.info(
        f"Precision: {precision_score(y_actual, y_pred, zero_division=0) * 100:.2f}%"
    )
    logger.info(f"Recall: {recall_score(y_actual, y_pred, zero_division=0) * 100:.2f}%")


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

    return train_df, test_df


def avg_yield_chg(
    given_date: str,
    simi_matrix_with_yield_chg,
    distance_min: float,
    distance_max: float,
    smooth_c: float = 5.0,
    yield_chg_fwd="yield_chg_fwd_10",
):
    """
    Calculate the weighted average yield change for a given date based on the similarity matrix.

    The weights are calculated as the inverse of the distance, meaning that smaller distances
    will have larger weights and larger distances will have smaller weights. The distances considered
    are those within the range specified by distance_min(included) and distance_max(NOT included).

    Args:
        given_date (str): Calculate other dates' weighted average yield change similar to this date.
        simi_matrix_with_yield_chg (DataFrame): The similarity matrix with yield changes.
        distance_min (float): included.
        distance_max (float): NOT included.
        smooth_c(float): smooth the weight differences caused by the inverse of the distance.
                        Weights are closer to distance average weights as smooth_c increases.
        yield_chg_fwd (str, optional): The column name for the forward yield change. Defaults to "yield_chg_fwd_10".

    Returns:
        yield_chg_avg(float): The weighted average yield change for the given row.
                            If no similar dates are found, return np.nan.
        close_rows(DataFrame): The rows in the similarity matrix that are within the specified distance range.
    """

    # Find the column with the same date as the given date
    close_rows = simi_matrix_with_yield_chg[["date", given_date, yield_chg_fwd]]
    # Find rows with distance in the specified range
    close_rows = close_rows[
        (close_rows[given_date] >= distance_min)
        & (close_rows[given_date] < distance_max)
    ]

    if close_rows.empty:
        return np.nan, close_rows
    # Calculate weighted average of yield_chg_fwd, weighted by INVERSE of distance(smaller distance, higher weight)
    weights = 1 / (close_rows[given_date] + smooth_c)
    weights = weights / weights.sum()
    yield_chg_avg = (close_rows[yield_chg_fwd] * weights).sum()

    return yield_chg_avg, close_rows


def predictions_test(
    simi_df: DataFrame,
    sample_df: DataFrame,
    distance_min: float,
    distance_max: float,
    smooth_c: float,
    train_ratio: float,
    gap: int,
):
    """
    Predict the yield change direction in n days and calculate the prediction accuracy of the test set.
    The sample_df will be split into training and testing sets based on the train_ratio and gap.
    The yield change direction is determined by yield_chg_fwd in the form of "yield_chg_fwd_n".
    n must be either 5, 10, 20, or 30. The prediction is based on the weighted average yield change
    for each row in the test DataFrame, where the weights are calculated based on the similarity matrix.

    Args:
        simi_df (DataFrame): Distance matrix, including column 'date'(m rows),
                                                        column ['2010-1-27', '2011-12-1'...](also m columns)
        sample_df (DataFrame): includes column 'date' and columns 'yield_chg_fwd_n'.
        distance_min (float): included.
        distance_max (float): NOT included
        smooth_c(float): smooth the weight differences caused by the inverse of the distance.
                    Weights are closer to distance average weights as smooth_c increases.
        train_ratio (float): The ratio of the data to be used for training. The rest will be used for testing.
        gap (int): The number of trade days between the end of training set and beginning of testing sets.
    Outputs:
        A test_predictions.csv file in the strategy_pool folder.
        Prints prediction statistics for each label in const.HistorySimilarity.LABELS.
    """

    # Add columns 'yield_chg_fwd_n' to simi_df.
    # simi_df includes column 'date'(m rows), column ['2010-1-27', '2011-12-1'...](also m columns)
    # plus n columns 'yield_chg_fwd_n'(depends on const.HistorySimilarity.LABELS).
    sample_df = sample_df[["date"] + list(const.HistorySimilarity.LABELS.values())]
    simi_df = pd.merge(sample_df, simi_df, left_on="date", right_on="date")
    train_df, test_df = train_test_split(sample_df, train_ratio=train_ratio, gap=gap)

    # Dates in history are used to calculate weighted average yield change.
    # Rows on test_df["date"] should NOT be included, since they are not in history.
    simi_df = simi_df[~simi_df["date"].isin(test_df["date"])]

    for day, name in const.HistorySimilarity.LABELS.items():
        test_df[f"pred_weight_{day}"] = test_df.apply(
            lambda row: (
                avg_yield_chg(
                    row["date"],
                    simi_df,
                    yield_chg_fwd=name,
                    distance_min=distance_min,
                    distance_max=distance_max,
                    smooth_c=smooth_c,
                )[0]
            ),
            axis=1,
        )
        # If weighted change > 0, prediction = 1;
        # if weighted change <= 0, prediction = 0;
        # else it cannot find similar dates within distance scope, then prediction = nan;
        test_df[f"pred_{day}"] = test_df[f"pred_weight_{day}"].apply(
            lambda x: 1 if x > 0 else 0 if x <= 0 else np.nan
        )
        test_df[f"actual_{day}"] = test_df[name].apply(
            lambda x: 1 if x > 0 else 0 if x <= 0 else np.nan
        )

        test_df = test_df.dropna(subset=[f"actual_{day}"])

    test_df.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, "test_predictions.csv"),
        encoding="utf-8",
        index=False,
    )

    # Make logger simple
    handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    logger.addHandler(handler)

    # Print train set and test set info
    logger.info(f"train size: {len(train_df)}; test size: {len(test_df)}")
    logger.info(
        f"train start date: {train_df['date'].iloc[0]}; train end date: {train_df['date'].iloc[-1]}"
    )
    logger.info(
        f"test start date: {test_df['date'].iloc[0]}; test end date: {test_df['date'].iloc[-1]}"
    )

    # Print prediction stats
    for day in const.HistorySimilarity.LABELS.keys():
        logger.info(f"###### Statics for Predicions of {day} days in the future ######")
        # Keep only rows where predictions are not NaN.
        # Nan predictions mean no similar dates are found in the history.
        valid_predictions = test_df[~test_df[f"pred_{day}"].isna()]
        logger.info(
            f"Valid_prediction_ratio: {len(valid_predictions) / len(test_df):.2f}"
        )
        test_df.dropna(subset=[f"pred_{day}", f"actual_{day}"])
        pred_values = test_df[f"pred_{day}"]
        actual_values = test_df.loc[pred_values.index, f"actual_{day}"]
        print_prediction_stats(actual_values, pred_values)


def predict_yield_chg(
    dates_to_pred: list,
    distance_min: float,
    distance_max: float,
    smooth_c: float,
) -> list:
    """
    Given some dates, predict the yield change direction in n days.
    1. Read a raw csv file(Path: const.PATH.STRATEGY_POOL/const.HistorySimilarity.SRC_NAME) ,
    which must include given dates.
    2. It will be processed to generate similarity matrix, features and yield change labels.
    3. The yield change direction is determined by yield_chg_fwd in the form of "yield_chg_fwd_n".
    n must be either 5, 10, 20, or 30. The prediction is based on the weighted average yield change
    of similar past dates, where the weights are calculated based on the similarity matrix.

    Args:
        dates_to_pred (list): A list of dates(str'%Y-%m-%d') to predict yield change.
        distance_min (float): included.
        distance_max (float): NOT included
        smooth_c(float): smooth the weight differences caused by the inverse of the distance.
                    Weights are closer to distance average weights as smooth_c increases.
    Returns:
        a list of DataFrames: DataFrames containing the predictions. One dataframe for each given date.
    """
    data = pd.read_csv(
        os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.SRC_NAME),
        parse_dates=["date"],
    )
    # history_df includes features and labels
    history_df = feature_engineering(
        df=data,
        yield_pctl_window=const.HistorySimilarity.PARAMS["YIELD_PCTL_WINDOW"],
        yield_chg_pctl_window=const.HistorySimilarity.PARAMS["YIELD_CHG_PCTL_WINDOW"],
        yield_chg_window_long=const.HistorySimilarity.PARAMS["YIELD_CHG_WINDOW_LONG"],
        yield_chg_window_short=const.HistorySimilarity.PARAMS["YIELD_CHG_WINDOW_SHORT"],
        stock_return_window=const.HistorySimilarity.PARAMS["STOCK_RETURN_WINDOW"],
        stock_return_pctl_window=const.HistorySimilarity.PARAMS[
            "STOCK_RETURN_PCTL_WINDOW"
        ],
        hs300_pctl_window=const.HistorySimilarity.PARAMS["HS300_PCTL_WINDOW"],
    )
    similarity_df = calculate_all_similarity(
        df=history_df,
        features=const.HistorySimilarity.FEATURES,
        metric="euclidean",
    )
    combined_df = history_df[["date"] + list(const.HistorySimilarity.LABELS.values())]
    # Change date type to string
    combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d')
    similarity_df['date'] = similarity_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

    combined_df = pd.merge(combined_df, similarity_df, left_on="date", right_on="date")

    # Predict only on given dates.
    result_df = combined_df[combined_df["date"].isin(dates_to_pred)]

    # Predict
    similar_dates = []
    for index, row in result_df.iterrows():
        # History predictions for similar dates with the same date (row["date"])
        similar_dates_with_one_date = pd.DataFrame()
        for day, label_name in const.HistorySimilarity.LABELS.items():
            yield_chg_avg, close_rows = avg_yield_chg(
                row["date"],
                combined_df,
                yield_chg_fwd=label_name,
                distance_min=distance_min,
                distance_max=distance_max,
                smooth_c=smooth_c,
            )
            result_df.at[index, f"pred_weight_{day}"] = yield_chg_avg
            result_df.at[index, f"pred_{day}"] = (
                1 if yield_chg_avg > 0 else 0 if yield_chg_avg <= 0 else np.nan
            )
            # Merge history predictions for similar dates into similar_dates_with_one_date(DataFrame)
            if similar_dates_with_one_date.empty:
                similar_dates_with_one_date = close_rows
            else:
                similar_dates_with_one_date = pd.merge(
                    similar_dates_with_one_date,
                    close_rows,
                    on=["date", row["date"]],
                    how="outer",
                )
        similar_dates.append(similar_dates_with_one_date)
    return similar_dates


if __name__ == "__main__":
    # MATRIX_EUCLIDEAN = f"similarity_matrix_euclidean.csv"
    # MATRIX_COSINE = f"similarity_matrix_cosine.csv"
    #
    # data_path = os.path.join(const.PATH.STRATEGY_POOL, "history_processed.csv")
    # all_samples = pd.read_csv(data_path)
    #
    # data_path = os.path.join(const.PATH.STRATEGY_POOL, MATRIX_EUCLIDEAN)
    # distance_df = pd.read_csv(data_path)
    #
    # predictions_test(
    #     simi_df=distance_df,
    #     sample_df=all_samples,
    #     distance_min=0,
    #     distance_max=1,
    #     smooth_c=2,
    #     train_ratio=0.9,
    #     gap=30,
    # )

    dates_to_predict = ["2024-4-17", "2024-4-16"]
    similar_dates_df = predict_yield_chg(
        dates_to_pred=dates_to_predict,
        distance_min=0.0,
        distance_max=1.0,
        smooth_c=2
    )
    for df in similar_dates_df:
        print(df)

