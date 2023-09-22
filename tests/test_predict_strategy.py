from fxincome.backtest.spread_backtrader import SpreadData, PredictStrategy
import datetime
import backtrader as bt
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from fxincome import logger, f_logger


class TestSpreadPortfolio:
    @pytest.fixture(scope="class")
    def global_data(self):
        global_data = {}
        global_data["leg1_code"] = "220210"
        global_data["leg2_code"] = "220215"
        global_data[
            "input_file"
        ] = f'./test_data/{global_data["leg1_code"]}_{global_data["leg2_code"]}_test_bt.csv'
        global_data["price_df"] = pd.read_csv(
            global_data["input_file"], parse_dates=["DATE"]
        )
        # minimum spread in the past 15 days are used as the threshold to open a position
        global_data["price_df"]["SPREAD_MIN"] = (
            global_data["price_df"]["SPREAD"].rolling(15).min()
        )
        global_data["price_df"].loc[:, ["SPREAD_MIN"]] = (
            global_data["price_df"].loc[:, ["SPREAD_MIN"]].fillna(method="backfill")
        )
        global_data["price_df"] = global_data["price_df"].dropna()
        return global_data

    def test_profit(self, global_data):
        cerebro = bt.Cerebro()
        data_long = SpreadData(dataname=global_data["price_df"], nocase=True)
        cerebro.adddata(data_long, name=PredictStrategy.ST_NAME)
        cerebro.addstrategy(PredictStrategy)
        cerebro.broker.set_cash(PredictStrategy.INIT_CASH)
        logger.info(global_data["leg1_code"] + "_" + global_data["leg2_code"])
        strategies = cerebro.run()
        profit = (cerebro.broker.get_value() - PredictStrategy.INIT_CASH) / 10000
        logger.info(
            f"PROFIT: {(cerebro.broker.get_value() - PredictStrategy.INIT_CASH) / 10000:.2f}"
        )
        assert profit == pytest.approx(-1.69, abs=0.01)
