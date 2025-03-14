# -*- coding: utf-8 -*-
import sqlite3

import pandas as pd
import os
import scipy
import datetime
from pandas import DataFrame
from fxincome import const, logger


def process_data():
    """
    Generate features, labels and similarity matrices.

    Returns:
        feats_labels(DataFrame): The feature_label DataFrame.
        euclidean_distance_df(DataFrame): The Euclidean distance matrix DataFrame.
        cosine_distance_df(DataFrame): The Cosine distance matrix DataFrame.
    """

    conn = sqlite3.connect(const.DB.SQLITE_CONN)
    raw_feature_table = const.DB.HistorySimilarity_TABLES["RAW_FEATURES"]
    feats_labels = pd.read_sql(
        f"SELECT * FROM [{raw_feature_table}]", conn, parse_dates=["date"]
    )

    feats_labels = feature_engineering(
        df=feats_labels,
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
    feats_labels.to_sql(
        const.DB.HistorySimilarity_TABLES["FEATS_LABELS"],
        conn,
        if_exists="replace",
        index=False,
    )

    conn.close()
    return feats_labels


if __name__ == "__main__":
    process_data()
