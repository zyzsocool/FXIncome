# -*- coding: utf-8; py-indent-offset:4 -*-

from datetime import timedelta
import math
import backtrader as bt
import pandas as pd
from analyzers.kelly import Kelly

class TBondData(bt.feeds.PandasData):
    # 对应OHLC净价的 OHLC YTM，在lines里分别命名如下
    lines = ('o_ytm', 'h_ytm', 'l_ytm', 'c_ytm')

    # 左边为lines里的名字，右边为dataframe column的名字
    params = (
        ('datetime', 'date'),
        ('open', 'b19_o'),
        ('high', 'b19_h'),
        ('low', 'b19_l'),
        ('close', 'b19_c'),
        ('volume', 'b19_amt'),
        ('openinterest', -1),
        ('o_ytm', 'b19_o_ytm'),
        ('h_ytm', 'b19_h_ytm'),
        ('l_ytm', 'b19_l_ytm'),
        ('c_ytm', 'b19_c_ytm')
    )


class PredictStrategy(bt.Strategy):
    tb_name = '019547.SH'  # name of the tbond

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.getdatabyname(self.tb_name).datetime.date(0)
        print(f"{dt:%Y%m%d} - {txt}")

    def __init__(self):
        # Keep references to the 'OHLC' and 'OHLC ytm' lines in the '019547.SH' dataseries
        self.open = self.getdatabyname(self.tb_name).open
        self.high = self.getdatabyname(self.tb_name).high
        self.low = self.getdatabyname(self.tb_name).low
        self.close = self.getdatabyname(self.tb_name).close
        self.o_ytm = self.getdatabyname(self.tb_name).o_ytm
        self.h_ytm = self.getdatabyname(self.tb_name).h_ytm
        self.l_ytm = self.getdatabyname(self.tb_name).l_ytm
        self.c_ytm = self.getdatabyname(self.tb_name).c_ytm
        # predictions of Ytm direction
        self.pred_df = pd.read_csv('d:/ProjectRicequant/fxincome/history_result.csv', parse_dates=['date'])
        # None means no pending order
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'Order {order.ref}, BUY EXECUTED, {order.executed.price:.2f}, {order.executed.size:.2f}')
            elif order.issell():
                self.log(f'Order {order.ref}, SELL EXECUTED, {order.executed.price:.2f}, {order.executed.size:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'Order {order.ref} Canceled/Margin/Rejected/Expired')

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"Trade Profit, Gross:{trade.pnl:.4f}, Net:{trade.pnlcomm:.4f}, Commission:{trade.commission:.4f}")


    def next(self):
        # Cancel the pending order if there is one
        self.cancel(self.order)
        today = self.getdatabyname(self.tb_name).datetime.datetime(0)
        preds = self.pred_df.query('date == @today')   # get ytm prediction for tomorrow
        if len(preds) == 0:  # Do nothing if no prediction
            return
        else:
            pred = int(preds.EnsembleVoteClassifier_pred.iat[0])  # get EnsembleVoteClassifier's prediction
        # valid until the t+2 bar's next()
        valid_day = self.getdatabyname(self.tb_name).datetime.datetime(1)

        if pred == 0:  # ytm down, price up, buy
            self.order = self.buy(
                data=self.getdatabyname(self.tb_name),
                size=self.__buy_all(self.close[0]),
                price=self.close[0],  # buy at today's close price
                exectype=bt.Order.Limit,
                valid=valid_day
            )
        elif pred == 1:  # ytm up, price down, sell
            self.order = self.sell(
                data=self.getdatabyname(self.tb_name),
                size=self.__sell_all(),
                price=self.close[0],  # sell at today's close price
                exectype=bt.Order.Limit,
                valid=valid_day
            )
        else:
            raise ValueError(f"The prediction is neither 0 nor 1")

    def __sell_size(self):
        """
        根据持仓情况生成卖出的数量，卖出数量不超过预先设定的值，且确保不会卖空。
            Returns:
                size(int): 应卖出的数量，大于或等于 0
        """
        size = 10
        position_size = self.getpositionbyname(self.tb_name).size
        if position_size <= 0:
            return 0
        elif position_size - size >= 0:
            return size
        else:
            return size - position_size

    def __sell_all(self):
        """
        卖出所有持仓，且确保不会卖空。
             Returns:
                size(int): 应卖出的数量，大于或等于 0
        """
        size = self.getpositionbyname(self.tb_name).size
        if size <= 0:
            return 0
        else:
            return size

    def __buy_size(self):
        """
        根据持仓情况生成买入的数量。
        """
        return 10

    def __buy_all(self, price):
        """
        用所有现金买入。
             Args:
                 price(float): 单价
             Returns:
                size(int): 应买入的数量，大于或等于 0
        """
        cash = self.broker.get_cash()
        if cash <= 0:
            return 0
        else:
            size = cash / price
            return math.floor(size)


def main():
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(PredictStrategy)

    price_df = pd.read_csv('d:/ProjectRicequant/fxincome/fxincome_features_latest.csv', parse_dates=['date'])
    # price_df = price_df[price_df['date'] < datetime.datetime(2021, 4, 15)]

    # Pass it to the backtrader datafeed and add it to the cerebro
    data1 = TBondData(dataname=price_df, nocase=True)
    cerebro.adddata(data1, name=PredictStrategy.tb_name)

    cerebro.broker.set_fundmode(True, 1)
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=0.0002, name=PredictStrategy.tb_name)  # commission is 0.02%
    #  Add analyzers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, fund=True, _name='TimeReturn')
    cerebro.addanalyzer(Kelly, _name='Kelly')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                        timeframe=bt.TimeFrame.Weeks,
                        riskfreerate=0.02,  # annual rate
                        fund=True,
                        _name='SharpRatio')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='SQN')
    cerebro.addanalyzer(bt.analyzers.DrawDown, fund=True, _name='DrawDown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='TradeReport')
    #  Add observers
    cerebro.addobserver(bt.observers.FundValue)
    cerebro.addobserver(bt.observers.FundShares)
    # Run over everything
    strategies = cerebro.run()

    # Plot the result
    # cerebro.plot(style='bar')

    broker = cerebro.broker
    print('Cash:' + str(broker.get_cash()))
    print('Value:' + str(broker.get_value()))
    print('fund value:' + str(broker.get_fundvalue()))
    print('fund share:' + str(broker.get_fundshares()))

    print(data1._name)
    position = cerebro.broker.getposition(data1)
    print(position)
    sharp_ratio = strategies[0].analyzers.SharpRatio
    sqn = strategies[0].analyzers.SQN
    draw_down = strategies[0].analyzers.DrawDown
    kelly  = strategies[0].analyzers.Kelly
    trade_report = strategies[0].analyzers.TradeReport
    sharp_ratio.print()
    sqn.print()
    draw_down.print()
    kelly.print()
    # trade_report.print()

if __name__ == '__main__':
    main()