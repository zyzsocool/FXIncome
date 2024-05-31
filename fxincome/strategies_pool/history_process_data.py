# -*- coding: utf-8 -*-
import pandas as pd
import os
from pandas import DataFrame
from fxincome import logger


def process(df: DataFrame,
            yield_percentile_window=5 * 250,
            yield_chg_window=5,
            yield_chg_percentile_window=5*250,
            stock_return_window=5):
    # Only bond trade days remain.
    df = df.dropna(subset=["t_chn_10y", "t_chn_1y"])
    # For missing values of us treasury bonds and hs300, fill them with the previous value.
    df.loc[:, ["t_usd_10y", "t_usd_1y", "hs300"]] = df[
        ["t_usd_10y", "t_usd_1y", "hs300"]
    ].ffill()
    df.loc[:, ["t_usd_chn_10y_spread"]] = df["t_usd_10y"] - df["t_chn_10y"]
    # 10y_yield_change = t_chn_10y(t) / t_chn_10y(t - yield_chg_window)
    df.loc[:, ["t_chn_10y_yield_chg"]] = df["t_chn_10y"] - df["t_chn_10y"].shift(
        yield_chg_window
    )
    # 1y_yield_change = t_chn_1y(t) / t_chn_1y(t - yield_chg_window)
    df.loc[:, ["t_chn_1y_yield_chg"]] = df["t_chn_1y"] - df["t_chn_1y"].shift(
        yield_chg_window
    )
    # stock_return = hs300(t) / hs300(t - stock_return_window)
    df.loc[:, ["stock_return"]] = df["hs300"] / df["hs300"].shift(
        stock_return_window
    )
    # Calculate the percentiles of 10-year Chinese Treasury bond yield, 1-year Chinese Treasury bond yield, 10-year US
    # Treasury and Chinese Treasury spread for the past "yield percentile window".
    df.loc[:, ["t_chn_10y_percentile"]] = (
        df["t_chn_10y"]
        .rolling(yield_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_chn_1y_percentile"]] = (
        df["t_chn_1y"]
        .rolling(yield_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_usd_chn_10y_spread_percentile"]] = (
        df["t_usd_chn_10y_spread"]
        .rolling(yield_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    # Calculate the percentiles of 10y_yield_change and 1y_yield_change for the past "yield change percentile window".
    df.loc[:, ["t_chn_10y_yield_chg_percentile"]] = (
        df["t_chn_10y_yield_chg"]
        .rolling(yield_chg_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_chn_1y_yield_chg_percentile"]] = (
        df["t_chn_1y_yield_chg"]
        .rolling(yield_chg_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )




    return df


if __name__ == "__main__":

    ROOT_PATH = "d:/ProjectRicequant/fxincome/strategies_pool"
    SRC_NAME = "history_similarity.csv"
    DEST_NAME = "history_processed.csv"

    YIELD_PERCENTILE_WINDOW = 5 * 250
    YIELD_CHG_PERCENTILE_WINDOW = 5 * 250
    YIELD_CHG_WINDOW = 5
    STOCK_RETURN_WINDOW = 5

    data = pd.read_csv(os.path.join(ROOT_PATH, SRC_NAME), parse_dates=["date"])
    data = process(df=data,
                   yield_percentile_window=YIELD_PERCENTILE_WINDOW,
                   yield_chg_percentile_window=YIELD_CHG_PERCENTILE_WINDOW,
                   yield_chg_window=YIELD_CHG_WINDOW,
                   stock_return_window=STOCK_RETURN_WINDOW)
    data.to_csv(os.path.join(ROOT_PATH, DEST_NAME), index=False, encoding="utf-8")
