# -*- coding: utf-8 -*-
import pandas as pd
import os
from pandas import DataFrame
from fxincome import const


def process(
    df: DataFrame,
    yield_percentile_window=5 * 250,
    yield_chg_window_long=20,
    yield_chg_window_short=10,
    yield_chg_percentile_window=5 * 250,
    stock_return_window=10,
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
    # Calculate the percentiles of 10-year Chinese Treasury bond yield, 1-year Chinese Treasury bond yield, 10-year US
    # Treasury and Chinese Treasury spread for the past "yield percentile window".
    df.loc[:, ["t_10y_percentile"]] = (
        df["t_10y"]
        .rolling(yield_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_1y_percentile"]] = (
        df["t_1y"]
        .rolling(yield_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_us_cn_10y_spread_percentile"]] = (
        df["t_us_cn_10y_spread"]
        .rolling(yield_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    # Calculate the percentiles of 10y_yield_change and 1y_yield_change for the past "yield change percentile window".
    df.loc[:, ["t_10y_yield_chg_long_percentile"]] = (
        df["t_10y_yield_chg_long"]
        .rolling(yield_chg_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_10y_yield_chg_short_percentile"]] = (
        df["t_10y_yield_chg_short"]
        .rolling(yield_chg_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    df.loc[:, ["t_1y_yield_chg_short_percentile"]] = (
        df["t_1y_yield_chg_short"]
        .rolling(yield_chg_percentile_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    # Generate new columns:
    # yield_chg_fwd_5, yield_chg_fwd_10, yield_chg_fwd_20, yield_chg_fwd_30.
    # yield_chg_fwd_n = t_10y(t+n) - t_10y(t)

    df = df.copy()
    for days in [5, 10, 20, 30]:
        df[f"yield_chg_fwd_{days}"] = df["t_10y"].shift(-days) - df["t_10y"]

    return df


if __name__ == "__main__":

    SRC_NAME = "history_similarity.csv"
    DEST_NAME = "history_processed.csv"

    YIELD_PERCENTILE_WINDOW = 5 * 250
    YIELD_CHG_PERCENTILE_WINDOW = 5 * 250
    YIELD_CHG_WINDOW_LONG = 20
    YIELD_CHG_WINDOW_SHORT = 10
    STOCK_RETURN_WINDOW = 10

    data = pd.read_csv(
        os.path.join(const.PATH.STRATEGY_POOL, SRC_NAME), parse_dates=["date"]
    )
    data = process(
        df=data,
        yield_percentile_window=YIELD_PERCENTILE_WINDOW,
        yield_chg_percentile_window=YIELD_CHG_PERCENTILE_WINDOW,
        yield_chg_window_long=YIELD_CHG_WINDOW_LONG,
        yield_chg_window_short=YIELD_CHG_WINDOW_SHORT,
        stock_return_window=STOCK_RETURN_WINDOW,
    )
    data.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, DEST_NAME), index=False, encoding="utf-8"
    )
