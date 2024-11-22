import pandas as pd
import os
import sqlite3
import datetime
import fxincome.strategies_pool.similarity_process_data as similarity_process_data
import fxincome.strategies_pool.similarity_model as similarity_model
import matplotlib.pyplot as plt

from pandas import DataFrame
from fxincome import logger, handler, const, data_handler
from WindPy import w


def update_data(distance_type: str):
    """
        Args:
        distance_type(str): The type of distance. Must be either "euclidean" or "cosine".
    Returns:
        feats_lables(DataFrame): The feature_label DataFrame.
        distance_df(DataFrame): The similarity matrix DataFrame.

    """
    if distance_type not in ["euclidean", "cosine"]:
        raise ValueError("distance_type must be either 'euclidean' or 'cosine'")

    # Update Raw Data to the latest date
    conn = sqlite3.connect(const.DB.SQLITE_CONN)
    w.start()
    data_handler.update_strat_hist_simi_raw_featrues(conn)
    asset_code = "511260.SH"
    data_handler.update_strat_hist_simi_raw_backtest(asset_code, conn)
    w.close()

    # Featrue Engineering
    raw_feature_table = const.DB.HistorySimilarity_TABLES["RAW_FEATURES"]
    features_from_db = pd.read_sql(
        f"SELECT * FROM [{raw_feature_table}]", conn, parse_dates=["date"]
    )
    feats_labels = similarity_process_data.feature_engineering(
        df=features_from_db,
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
    distance_df = similarity_process_data.calculate_all_similarity(
        feats_labels, const.HistorySimilarity.FEATURES, metric=distance_type
    )
    if distance_type == "euclidean":
        simi_file = const.HistorySimilarity.SIMI_EUCLIDEAN
    else:
        simi_file = const.HistorySimilarity.SIMI_COSINE
    distance_df.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, simi_file),
        encoding="utf-8",
        index=False,
    )
    conn.close()
    return feats_labels, distance_df


def predict_everyday(
    feats_labels: DataFrame,
    distance_df: DataFrame,
    pred_days: int = 5,
    distance_min: float = 0.00,
    distance_max: float = 0.25,
    smooth_c: int = 5,
):
    last_date = feats_labels.date.max()
    logger.info(f"Last date: {last_date}")
    predictions, similar_dates_dict = similarity_model.predict_yield_chg(
        dates_to_pred=[last_date],
        similarity_df=distance_df,
        sample_df=feats_labels,
        distance_min=distance_min,
        distance_max=distance_max,
        smooth_c=smooth_c,
    )
    yield_chg_pred = predictions[f"pred_{pred_days}"].iloc[0]
    similar_dates_df = similar_dates_dict[last_date]
    similar_dates_df = similar_dates_df.rename(
        columns={last_date: "distance"}
    )
    similar_dates_df = similar_dates_df[
        ["date", "distance", f"yield_chg_fwd_{pred_days}"]
    ]

    print(
        f"10-Year-Yield change in {pred_days} trade days will be: {'UP' if yield_chg_pred > 0 else 'DOWN'}"
    )
    print(similar_dates_df)
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
    ytm_df['date'] = ytm_df['date'].dt.date
    ytm_df = ytm_df.sort_values(by="date").reset_index(drop=True)

    # Select the 10 most similar dates based on the smallest distances
    most_similar_dates = similar_dates_df.sort_values(by="distance").head(10)['date']

    # Extract ytms for each date in latest similar dates
    ytm_data = {}
    for date in most_similar_dates:
        start_idx = ytm_df[ytm_df['date'] == date].index[0]
        end_idx = start_idx + pred_days
        ytm_data[date] = ytm_df.loc[start_idx:end_idx, 't_10y'].values

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
    # Set parameters
    distance_type = "euclidean"
    pred_days = 5
    distance_min = 0.0
    distance_max = 0.25
    smooth_c = 5

    # Update data from Wind and calculate similarity
    feats_labels, distance_df = update_data(distance_type)

    # Predict yield change
    similar_dates_df = predict_everyday(
        feats_labels=feats_labels,
        distance_df=distance_df,
        pred_days=pred_days,
        distance_min=distance_min,
        distance_max=distance_max,
        smooth_c=smooth_c,
    )

    # Plot YTM trends of similar dates in history
    plot_ytms(similar_dates_df, pred_days)

if __name__ == "__main__":
    main()
