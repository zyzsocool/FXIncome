# -*- coding: utf-8; py-indent-offset:4 -*-
import os
import math
import datetime
import backtrader as bt
import numpy as np
import pandas as pd
from analyzers.kelly import Kelly
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from dataclasses import dataclass
from fxincome import logger, handler, const


class ETFData(bt.feeds.PandasData):
    lines = ("turnover", "gc001")

    # 左边为lines里的名字，右边为dataframe column的名字
    params = (
        ("datetime", "date"),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),  # 成交额，单位是元
        ("openinterest", -1),
        ("turnover", "turnover"),  # 换手率，单位是%
        ("gc001", "GC001_close"),  # GC001 回购利率收盘价
    )


@dataclass
class Trader:
    judgement_day: datetime.date
    cash: float
    bond_commission_rate: float
    repo_commission_rate: float
    position: int = 0

    def buy_all(self, price: float) -> int:
        """
        Calculate the size to buy with all cash, accounting for commission, and update cash and position.
        Args:
            price(float): The price to buy

        Returns:
            size(int): The size to buy, greater than or equal to 0
        """
        size = math.floor(self.cash / (price * (1 + self.bond_commission_rate)))
        total_cost = size * price * (1 + self.bond_commission_rate)
        self.cash = self.cash - total_cost
        self.position = self.position + size
        return size

    def sell_all(self, price: float) -> int:
        """
        Calculate the cash received from selling all positions, accounting for commission, and update cash and position.
        Args:
            price(float): The price to sell

        Returns:
            size(int): The size to sell, greater than or equal to 0
        """
        cash_received = self.position * price * (1 - self.bond_commission_rate)
        self.cash = self.cash + cash_received
        size = self.position
        self.position = 0
        return size

    def reverse_repo(
        self, rate: float, start_date: datetime.date, end_date: datetime.date
    ):
        """
        Use all remaining cash to do reverse repo. Calculate the interest received from reverse repo, and update cash.
        Args:
            rate(float): The annualized interest rate. 0.01 for 1%.
            start_date(datetime.date): The start date of reverse repo
            end_date(datetime.date): The end date of reverse repo
        """
        days = (end_date - start_date).days
        interest = self.cash * rate * days / 365
        commission = self.cash * self.repo_commission_rate
        self.cash = self.cash + interest - commission


class NTraderStrategy(bt.Strategy):
    """
    Cash are divided into n parts, and each part is used to trade on a different day based on predictions.
    num_traders <= pred_days.
    Parmas can be accessed by self.p.xxx
        num_traders: Number of traders
        pred_days: YTM direction = ytm(t + pred_days) - ytm(t)
        pred_df: DataFrame with columns ["date", "pred_5", "pred_10", ..., "pred_{n}"]. n = pred_days
    """

    params = dict(
        num_traders=1,
        pred_days=10,
        pred_df=None,
    )

    etf_name = "511260.SH"  # 国泰上证10年期国债ETF

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.etf_data.datetime.date(0)
        print(f"{dt:%Y%m%d} - {txt}")

    def total_value(self):
        cash = sum([trader.cash for trader in self.traders])
        position = sum([trader.position for trader in self.traders])
        return cash + position * self.etf_data.close[0]

    def __init__(self):
        self.etf_data = self.getdatabyname(self.etf_name)

        # None means no pending order
        self.order = None

        total_days = self.etf_data.buflen() - 1

        commission_rate = self.broker.getcommissioninfo(self.etf_data).p.commission

        self.traders = [
            Trader(
                # The first n judgement days are the first n days of backtest data.
                judgement_day=self.etf_data.datetime.date(-total_days + n),
                cash=self.broker.get_cash() / self.p.num_traders,
                position=0,
                bond_commission_rate=commission_rate,
                repo_commission_rate=0.001 / 100,
            )
            for n in range(self.p.num_traders)
        ]

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"Order {order.ref}, BUY, {order.executed.price:.2f}, {order.executed.size:.2f}"
                )
            elif order.issell():
                self.log(
                    f"Order {order.ref}, SELL, {order.executed.price:.2f}, {order.executed.size:.2f}"
                )
        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected,
            order.Expired,
        ]:
            self.log(f"Order {order.ref} Canceled/Margin/Rejected/Expired")

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(
                f"Trade Profit, Gross:{trade.pnl:.4f}, Net:{trade.pnlcomm:.4f}, Commission:{trade.commission:.4f}"
            )

    def next(self):
        today = self.etf_data.datetime.date(0)

        for trader in self.traders:
            # Do GC001 if today is not the last day of backtest data.
            if len(self.etf_data) + 1 <= self.etf_data.buflen():
                trader.reverse_repo(
                    rate=self.etf_data.gc001[0] / 100,
                    start_date=today,
                    end_date=self.etf_data.datetime.date(1),
                )
            # Only on the judgement_day we decide whether to trade.
            if trader.judgement_day != today:
                continue
            # Update the judgement day to pred_days later
            if len(self.etf_data) + self.p.pred_days <= self.etf_data.buflen():
                trader.judgement_day = self.etf_data.datetime.date(self.p.pred_days)
            else:
                continue

            # Trade based on the predictions
            preds = self.p.pred_df[self.p.pred_df["date"].dt.date == today]

            if len(preds) == 0:  # Do nothing if no prediction
                logger.info(f"No prediction")
                continue
            else:
                # prediction of ytm direction in next pred_days trade days.
                pred = preds[f"pred_{self.p.pred_days}"].iat[0]
            if pred == 0:  # ytm down, buy
                self.order = self.buy(
                    data=self.etf_data,
                    size=trader.buy_all(
                        self.etf_data.open[1]
                    ),  # buy at t+1's open price
                    price=None,  # buy at t+1's open price
                    exectype=bt.Order.Market,  # buy at t+1's open price
                    valid=None,
                )
            elif pred == 1:  # ytm up, sell
                self.order = self.sell(
                    data=self.etf_data,
                    size=trader.sell_all(
                        self.etf_data.open[1]
                    ),  # sell at t+1's open price
                    price=None,  # sell at t+1's open price
                    exectype=bt.Order.Market,  # sell at t+1's open price
                    valid=None,
                )
            else:
                logger.info(f"The prediction is neither 0 nor 1")
                continue


def run_backtest(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    num_traders: int,
    pred_days: int,
):
    bond_pred, etf_price = read_predictions_prices(start_date, end_date)

    # Create a cerebro entity
    cerebro = bt.Cerebro(tradehistory=True)

    # Add Strategies
    cerebro.addstrategy(
        NTraderStrategy, num_traders=num_traders, pred_days=pred_days, pred_df=bond_pred
    )

    # Add data to Cerebro
    data1 = ETFData(dataname=etf_price, nocase=True)
    cerebro.adddata(data1, name=NTraderStrategy.etf_name)

    cerebro.broker.set_fundmode(True, 1)
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(
        commission=0.0002, name=NTraderStrategy.etf_name
    )  # commission is 0.02%

    #  Add analyzers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, fund=True, _name="TimeReturn")
    cerebro.addanalyzer(Kelly, _name="Kelly")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        timeframe=bt.TimeFrame.Weeks,
        riskfreerate=0.018,  # annual rate
        fund=True,
        _name="SharpRatio",
    )
    cerebro.addanalyzer(bt.analyzers.SQN, _name="SQN")
    cerebro.addanalyzer(bt.analyzers.DrawDown, fund=True, _name="DrawDown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="TradeReport")
    #  Add observers
    cerebro.addobserver(bt.observers.FundValue)
    cerebro.addobserver(bt.observers.FundShares)
    # Run over everything
    strategies = cerebro.run()
    strategy = strategies[0]
    print_backtest_result(cerebro, data1, strategy)
    # for strategy in strategies:
    #     print_result(cerebro, data1, strategy)


def print_backtest_result(cerebro, data1, strategy):
    # Plot the result
    # cerebro.plot(style="bar")
    broker = cerebro.broker
    print(data1._name)

    traders_cash = sum([trader.cash for trader in strategy.traders])
    traders_position = sum([trader.position for trader in strategy.traders])

    print("Broker Cash:" + str(broker.get_cash()))
    print("Traders Cash:" + str(traders_cash))
    print(cerebro.broker.getposition(data1))
    print("Traders Position:" + str(traders_position))

    print(f"Strategy Total Value: {strategy.total_value()}")
    print("Broker Value:" + str(broker.get_value()))
    print("Fund value:" + str(broker.get_fundvalue()))
    print("Fund share:" + str(broker.get_fundshares()))

    sharp_ratio = strategy.analyzers.SharpRatio
    sqn = strategy.analyzers.SQN
    draw_down = strategy.analyzers.DrawDown
    kelly = strategy.analyzers.Kelly
    trade_report = strategy.analyzers.TradeReport
    sharp_ratio.print()
    # sqn.print()
    # draw_down.print()
    # kelly.print()
    # trade_report.print()


def read_predictions_prices(
    start_date: datetime.datetime, end_date: datetime.datetime
) -> tuple:
    # predictions of Ytm direction
    bond_pred = pd.read_csv(
        os.path.join(const.PATH.STRATEGY_POOL, const.HistorySimilarity.PREDICT_FILE),
        parse_dates=["date"],
    )
    etf_price = pd.read_csv(
        os.path.join(const.PATH.STRATEGY_POOL, "511260.sh.csv"),
        parse_dates=["date"],
    )
    # Filter prices between start_date and end_date
    etf_price = etf_price[
        (etf_price["date"] >= start_date) & (etf_price["date"] <= end_date)
    ]
    numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
    for col in numeric_cols:
        etf_price[col] = pd.to_numeric(etf_price[col])

    return bond_pred, etf_price


def analyze_prediction(
    start_date: datetime.datetime, end_date: datetime.datetime, pred_days: int
):
    bond_pred, etf_price = read_predictions_prices(start_date, end_date)

    tbond_df = pd.read_csv(
        os.path.join(const.PATH.STRATEGY_POOL, "history_processed.csv"),
        parse_dates=["date"],
    )
    tbond_df = tbond_df[["date", "t_10y"]]
    etf_price = pd.merge(etf_price, tbond_df, on="date", how="left")
    etf_price[f"y_fwd_{pred_days}"] = (
        etf_price["t_10y"].shift(-pred_days) - etf_price["t_10y"]
    )
    etf_price[f"y_actual_{pred_days}"] = etf_price[f"y_fwd_{pred_days}"].apply(
        lambda x: 1 if x > 0 else 0
    )

    bond_pred[f"actual_{pred_days}"] = bond_pred[f"yield_chg_fwd_{pred_days}"].apply(
        lambda x: 1 if x > 0 else 0
    )

    etf_price[f"price_chg_{pred_days}"] = (
        etf_price["close"].shift(-pred_days) - etf_price["close"]
    )
    etf_price[f"actual_price_{pred_days}"] = etf_price[f"price_chg_{pred_days}"].apply(
        lambda x: 1 if x > 0 else 0
    )

    merged_df = pd.merge(etf_price, bond_pred, on="date", how="left")
    merged_df = merged_df.dropna()

    bond_pred_values = merged_df[f"pred_{pred_days}"]
    bond_actual_values = merged_df[f"actual_{pred_days}"]
    etf_actual_values = merged_df[f"actual_price_{pred_days}"]
    y_actual_values = merged_df[f"y_actual_{pred_days}"]

    bond_pred_accuracy = accuracy_score(bond_actual_values, bond_pred_values)
    bond_pred_etf_accuracy = accuracy_score(etf_actual_values, 1 - bond_pred_values)
    bond_actual_etf_accuracy = accuracy_score(etf_actual_values, 1 - bond_actual_values)
    yield_actual_etf_accuracy = accuracy_score(etf_actual_values, 1 - y_actual_values)

    logger.info(f"bond_pred_accuracy: {bond_pred_accuracy}")
    logger.info(f"bond_pred_etf_accuracy: {bond_pred_etf_accuracy}")
    logger.info(f"bond_actual_etf_accuracy: {bond_actual_etf_accuracy}")
    logger.info(f"yield_actual_etf_accuracy: {yield_actual_etf_accuracy}")
    print(confusion_matrix(etf_actual_values, 1 - bond_actual_values))
    print(confusion_matrix(etf_actual_values, 1 - y_actual_values))

    merged_df.to_csv(
        os.path.join(const.PATH.STRATEGY_POOL, "bond_etf_predictions.csv"), index=False
    )


def main():
    start_date = datetime.datetime(2022, 1, 1)
    end_date = datetime.datetime(2024, 5, 30)
    run_backtest(start_date, end_date, num_traders=10, pred_days=10)
    # analyze_prediction(start_date, end_date, pred_days=10)


if __name__ == "__main__":
    main()
