# encoding=utf-8
import datetime
import sqlite3
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
import fxincome.strategies_pool.similarity_process_data as similarity_process_data

from pandas import DataFrame
from dataclasses import dataclass
from WindPy import w
from fxincome import const, logger, data_handler
from fxincome.strategies_pool.similarity_model import predict_yield_chg
from fxincome.backtest.similarity_backtest import NTraderStrategy, run_backtest


@dataclass
class Portfolio:
    name: str
    asset_code: str
    distance_type: str
    strat: bt.Strategy
    num_traders: int
    pred_days: int
    distance_min: float = 0.00
    distance_max: float = 0.25
    smooth_c: int = 5

    def __post_init__(self):
        if self.distance_type not in ["euclidean", "cosine"]:
            raise ValueError("distance_type must be either 'euclidean' or 'cosine'")

    def __str__(self):
        return(
            f"Portfolio: {self.name}, Strategy: {self.strat.__name__}\n"
            f"Asset Code: {self.asset_code}, Distance Type: {self.distance_type}\n"
            f"Number of Traders: {self.num_traders}, Prediction Days Forwad: {self.pred_days}\n"
            f"Distance Scope: [{self.distance_min}, {self.distance_min}], Smooth C: {self.smooth_c}\n"
        )

def update_data(asset_code_set: set, strat_date: datetime.date):
    # Update Raw Data to the latest date
    conn = sqlite3.connect(const.DB.SQLITE_CONN)
    w.start()
    data_handler.update_strat_hist_simi_raw_featrues(conn)
    for asset_code in asset_code_set:
        data_handler.update_strat_hist_simi_raw_backtest(asset_code, conn)
    w.close()

    # Update Real Trade Price
    for asset_code in asset_code_set:
        raw_backtest_table = const.DB.HistorySimilarity_TABLES["RAW_BACKTEST"]
        sql = f"SELECT * FROM [{raw_backtest_table}] where asset_code='{asset_code}' and date>='{strat_date}'"
        raw_df = pd.read_sql(sql, conn)
        real_backtest_table = const.DB.HistorySimilarity_TABLES["REAL_BACKTEST"]
        sql = f"SELECT date FROM [{real_backtest_table}] where asset_code='{asset_code}' and date>='{strat_date}'"
        real_dates = pd.read_sql(sql, conn)["date"].tolist()
        append_data = []
        for row, data in raw_df.iterrows():
            if data["date"] in real_dates:
                continue
            else:
                data["open"] = input(
                    f"输入{data['date']}日，{data['asset_code']}可以交易的开盘价(净价):"
                )
                data["asset_code"] = asset_code
                append_data.append(data)
        if not append_data:
            continue
        append_df = pd.DataFrame(append_data)
        print(append_df)
        choice = input("是否确认添加数据？(y/n)")
        if choice == "y":
            append_df.to_sql(
                const.DB.HistorySimilarity_TABLES["REAL_BACKTEST"],
                conn,
                if_exists="append",
                index=False,
            )
        else:
            print("已取消添加")

        conn.close()


def predict_everyday(portfolio: Portfolio, feats_labels: DataFrame, distance_df: DataFrame):
    conn = sqlite3.connect(const.DB.SQLITE_CONN)
    old_predictions = pd.read_sql(
        f"SELECT * FROM 'strat.hist_simi.temp.predictions_{portfolio.name}' ",
        conn,
        parse_dates=["date"],
    )
    start_date = old_predictions["date"].max()
    end_date = datetime.date.today()

    dates_to_predict = [d.date() for d in pd.date_range(start_date, end_date)]
    predictions, similar_dates_dict = predict_yield_chg(
        dates_to_pred=dates_to_predict,
        similarity_df=distance_df,
        sample_df=feats_labels,
        distance_min=portfolio.distance_min,
        distance_max=portfolio.distance_max,
        smooth_c=portfolio.smooth_c,
    )

    # Retrieve the most similar dates for the last date
    last_date = predictions["date"].max()
    similar_dates_df = similar_dates_dict[last_date]
    similar_dates_df = similar_dates_df.rename(columns={last_date: "distance"})
    similar_dates_df = similar_dates_df[
        ["date", "distance", f"yield_chg_fwd_{portfolio.pred_days}"]
    ]

    if len(predictions) <= 1:
        logger.info(f"{portfolio.name}: No predictions to update.")
        conn.close()
        return similar_dates_df
    predictions[1:].to_sql(
        f"strat.hist_simi.temp.predictions_{portfolio.name}",
        conn,
        if_exists="append",
        index=False,
    )
    conn.close()
    return similar_dates_df


def plot_ytms(similar_dates_df, pred_days):
    conn = sqlite3.connect(const.DB.SQLITE_CONN)
    ytm_df = pd.read_sql(
        f"SELECT date, t_10y FROM [{const.DB.HistorySimilarity_TABLES['RAW_FEATURES']}]",
        conn,
        parse_dates=["date"],
    )
    conn.close()
    ytm_df = ytm_df.dropna()
    ytm_df["date"] = ytm_df["date"].dt.date
    ytm_df = ytm_df.sort_values(by="date").reset_index(drop=True)

    # Select the 10 most similar dates based on the smallest distances
    most_similar_dates = similar_dates_df.sort_values(by="distance").head(10)["date"]

    # Extract ytms for each date in latest similar dates
    ytm_data = {}
    for date in most_similar_dates:
        start_idx = ytm_df[ytm_df["date"] == date].index[0]
        end_idx = start_idx + pred_days
        ytm_data[date] = ytm_df.loc[start_idx:end_idx, "t_10y"].values

    # Plot ytms for each date in latest similar dates
    plt.figure(figsize=(5, 10))
    for date, ytms in ytm_data.items():
        plt.plot(range(pred_days + 1), ytms, label=f"Date: {date}")

    # Combine plots into a single diagram
    plt.xlabel("Days")
    plt.ylabel("10y YTM")
    plt.title("10-Year YTM Over Time in Most Similar Dates")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    start_date = datetime.date(2024, 11, 11)
    end_date = datetime.date.today()
    portfolios = [
        Portfolio("p1", "240004.IB", "euclidean", NTraderStrategy, 10, 10),
        Portfolio("p2", "240004.IB", "cosine", NTraderStrategy, 10, 10),
        Portfolio("p3", "240004.IB", "euclidean", NTraderStrategy, 5, 5),
        Portfolio("p4", "240004.IB", "cosine", NTraderStrategy, 5, 5),
    ]
    update_data({pf.asset_code for pf in portfolios}, start_date)
    feats_labels, euclidean_distances, cosine_distances = similarity_process_data.process_data()
    for pf in portfolios:
        if pf.distance_type == "euclidean":
            distance_df = euclidean_distances
        else:
            distance_df = cosine_distances
        similar_dates_df = predict_everyday(pf, feats_labels, distance_df)
        plot_ytms(similar_dates_df, pf.pred_days)
        print(pf)
        print(
            f"\033[91m========================Running backtest for {pf.name}==================================\033[0m"
        )
        run_backtest(
            strat=pf.strat,
            asset_code=pf.asset_code,
            start_date=start_date,
            end_date=end_date,
            num_traders=pf.num_traders,
            pred_days=pf.pred_days,
            pred_table=f"strat.hist_simi.temp.predictions_{pf.name}",
            etf_table=const.DB.HistorySimilarity_TABLES["REAL_BACKTEST"],
            sizer="all",
            tp_pct=0.0015,
            sl_pct=-0.0008,
            repo_commission=0.001 / 100,
            bond_commission=0.0002,
        )


if __name__ == "__main__":
    main()
