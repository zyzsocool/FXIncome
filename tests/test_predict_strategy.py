from fxincome.backtest.spread_backtrader import SpreadData, PredictStrategy
import backtrader as bt
import pandas as pd
import pytest
from fxincome.const import PATH, SPREAD
from fxincome import logger, f_logger


def main():
    leg1_code = "220210"
    leg2_code = "220215"
    cerebro = bt.Cerebro()
    input_file = PATH.SPREAD_DATA + leg1_code + "_" + leg2_code + "_test_bt.csv"
    price_df = pd.read_csv(input_file, parse_dates=["DATE"])
    # minimum spread in the past 15 days are used as the threshold to open a position
    price_df["SPREAD_MIN"] = price_df["SPREAD"].rolling(15).min()
    price_df.loc[:, ["SPREAD_MIN"]] = price_df.loc[:, ["SPREAD_MIN"]].fillna(
        method="backfill"
    )
    price_df = price_df.dropna()
    # data_long and data_short are the same data feeds, but with different names.
    # data_long is for predicting long strategy, and data_short is for predicting short strategy.
    data_long = SpreadData(dataname=price_df, nocase=True)
    cerebro.adddata(data_long, name=PredictStrategy.ST_NAME)
    cerebro.addstrategy(PredictStrategy)
    cerebro.broker.set_cash(PredictStrategy.INIT_CASH)
    logger.info(leg1_code + "_" + leg2_code)
    strategies = cerebro.run()
    profit = (cerebro.broker.get_value() - PredictStrategy.INIT_CASH) / 10000
    logger.info(
        f"PROFIT: {(cerebro.broker.get_value() - PredictStrategy.INIT_CASH) / 10000:.2f}"
    )
    assert profit == pytest.approx(-1.69, abs=0.01)


if __name__ == "__main__":
    main()
