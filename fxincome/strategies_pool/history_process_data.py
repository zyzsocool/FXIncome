# -*- coding: utf-8 -*-
import pandas as pd
import os
import scipy
from pandas import DataFrame
from fxincome import const


def feature_engineering(
    df: DataFrame,
    yield_pctl_window=5 * 250,
    yield_chg_window_long=20,
    yield_chg_window_short=10,
    yield_chg_pctl_window=5 * 250,
    stock_return_window=10,
    stock_return_pctl_window=5 * 250,
    hs300_pctl_window=5 * 250,
):
    # Only bond trade days remain.
    df = df.dropna(subset=["t_10y", "t_1y"])
    # For missing values of us treasury bonds and hs300, fill them with the previous value.
    df.loc[:, ["t_us_10y", "t_us_1y", "hs300"]] = df[
        ["t_us_10y", "t_us_1y", "hs300"]
    ].ffill()
    df.loc[:, ["t_us_cn_10y_spread"]] = df["t_us_10y"] - df["t_10y"]

    # 10-year Chinese Treasury bond yield change are calculated by short and long term.
    # 10y_yield_change_long = t_10y(t) / t_10y(t - yield_chg_window_long)
    # 10y_yield_change_short = t_10y(t) / t_10y(t - yield_chg_window_short)
    # 1y_yield_change_short = t_1y(t) / t_1y(t - yield_chg_window_short)
    df.loc[:, ["t_10y_yield_chg_long"]] = df["t_10y"] - df["t_10y"].shift(
        yield_chg_window_long
    )
    df.loc[:, ["t_10y_yield_chg_short"]] = df["t_10y"] - df["t_10y"].shift(
        yield_chg_window_short
    )
    df.loc[:, ["t_1y_yield_chg_short"]] = df["t_1y"] - df["t_1y"].shift(
        yield_chg_window_short
    )

    # stock_return = hs300(t) / hs300(t - stock_return_window) - 1
    df.loc[:, ["stock_return"]] = (
        df["hs300"] / df["hs300"].shift(stock_return_window) - 1
    )
    df.loc[:, ["stock_return_pctl"]] = (
        df["stock_return"]
        .rolling(stock_return_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["hs300_pctl"]] = (
        df["hs300"]
        .rolling(hs300_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    # Calculate the percentiles of 10-year Chinese Treasury bond yield, 1-year Chinese Treasury bond yield, 10-year US
    # Treasury and Chinese Treasury spread for the past "yield percentile window".
    df.loc[:, ["t_10y_pctl"]] = (
        df["t_10y"]
        .rolling(yield_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_1y_pctl"]] = (
        df["t_1y"]
        .rolling(yield_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_us_cn_10y_spread_pctl"]] = (
        df["t_us_cn_10y_spread"]
        .rolling(yield_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    # Calculate the percentiles of 10y_yield_change and 1y_yield_change for the past "yield change percentile window".
    df.loc[:, ["t_10y_yield_chg_long_pctl"]] = (
        df["t_10y_yield_chg_long"]
        .rolling(yield_chg_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_10y_yield_chg_short_pctl"]] = (
        df["t_10y_yield_chg_short"]
        .rolling(yield_chg_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_1y_yield_chg_short_pctl"]] = (
        df["t_1y_yield_chg_short"]
        .rolling(yield_chg_pctl_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    # yield_chg_fwd_n = t_10y(t+n) - t_10y(t)
    df = df.copy()
    # iterate const.HistorySimilarity.LABELS to generate label values.
    for day, name in const.HistorySimilarity.LABELS.items():
        df[name] = df["t_10y"].shift(-day) - df["t_10y"]

    return df


def calculate_all_similarity(df, features: list, metric="euclidean"):
    """
    Calculate the Euclidean distance between each pair of rows in a DataFrame for the given features.

    Args:
        df (DataFrame): The input DataFrame containing the data. It must include a 'date' column.
        features (list): The list of feature column names to be used in the distance calculation.
        metric (str): The type of similarity. Must be either "euclidean" or "cosine".
    Returns:
        similarity_matrix(DataFrame): A n * n matrix.
                         Index and columns are the dates, and the values are the distances between index and dates.
    """
    if metric not in ["euclidean", "cosine"]:
        raise ValueError("sim_type must be either 'euclidean' or 'cosine'")

    # Separate dates from features
    dates = df["date"].dt.date
    feature_data = df[features].values

    # The result is a 2D array where the element at position (i, j) is
    # the distance between the i-th and j-th row of the DataFrame
    distances = scipy.spatial.distance.cdist(feature_data, feature_data, metric=metric)

    # Create a DataFrame where index and columns are the dates, and the values are the distances
    similarity_matrix = pd.DataFrame(distances, index=dates, columns=dates)

    return similarity_matrix


if __name__ == "__main__":

    YIELD_PCTL_WINDOW = 5 * 250
    YIELD_CHG_PCTL_WINDOW = 5 * 250
    YIELD_CHG_WINDOW_LONG = 20
    YIELD_CHG_WINDOW_SHORT = 10
    STOCK_RETURN_WINDOW = 10
    STOCK_RETURN_PCTL_WINDOW = 5 * 250
    HS300_PCTL_WINDOW = 5 * 250

    data = pd.read_csv(
        os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.SRC_NAME),
        parse_dates=["date"],
    )
    data = feature_engineering(
        df=data,
        yield_pctl_window=YIELD_PCTL_WINDOW,
        yield_chg_pctl_window=YIELD_CHG_PCTL_WINDOW,
        yield_chg_window_long=YIELD_CHG_WINDOW_LONG,
        yield_chg_window_short=YIELD_CHG_WINDOW_SHORT,
        stock_return_window=STOCK_RETURN_WINDOW,
        stock_return_pctl_window=STOCK_RETURN_PCTL_WINDOW,
        hs300_pctl_window=HS300_PCTL_WINDOW,
    )
    data.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.FEATURE_FILE),
        index=False,
        encoding="utf-8",
    )

    data = data.dropna().reset_index(drop=True)

    distance_df = calculate_all_similarity(
        data, const.HistorySimilarity.FEATURES, metric="euclidean"
    )

    distance_df.reset_index().rename(columns={'index': 'date'}).to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.SIMI_EUCLIDEAN),
        encoding="utf-8",
        index=False
    )

    distance_df = calculate_all_similarity(
        data, const.HistorySimilarity.FEATURES, metric="cosine"
    )

    distance_df.reset_index().rename(columns={'index': 'date'}).to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.SIMI_COSINE),
        encoding="utf-8",
        index=False
    )
