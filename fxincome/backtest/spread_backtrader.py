import datetime
import backtrader as bt
import pandas as pd
from fxincome.const import PATH, SPREAD


class SpreadData(bt.feeds.PandasData):
    # 对应OHLC净价的 OHLC YTM，在lines里分别命名如下
    lines = (
        'code_leg1', 'out_leg1', 'ytm_leg1', 'lend_rate_leg1', 'code_leg2',
        'vol_leg2', 'out_leg2', 'ytm_leg2', 'lend_rate_leg2', 'spread', 'spread_min',)

    params = (('datetime', 'DATE'), ('close', 'CLOSE'), ('open', 'OPEN'), ('high', 'HIGH'), ('low', 'LOW'),
              ('volume', 'VOL_LEG1'), ('openinterest', -1), ('code_leg1', 'CODE_LEG1'),
              ('out_leg1', 'OUT_BAL_LEG1'), ('ytm_leg1', 'YTM_LEG1'), ('lend_rate_leg1', 'LEND_RATE_LEG1'),
              ('code_leg2', 'CODE_LEG2'), ('vol_leg2', 'VOL_LEG2'),
              ('out_leg2', 'OUT_BAL_LEG2'), ('ytm_leg2', 'YTM_LEG2'), ('lend_rate_leg2', 'LEND_RATE_LEG2'),
              ('spread', 'SPREAD'), ('spread_min', 'SPREAD_MIN'),)


class SpreadBaselineStrategy(bt.Strategy):
    st_name = 'baseline'
    lend_fee = 0.0

    def log(self, txt, dt=None):
        """ Logging function for this strategy"""
        dt = dt or self.getdatabyname(self.st_name).datetime.date(0)
        print(f"{dt:%Y%m%d} - {txt}")

    def __init__(self):
        self.spread = self.getdatabyname(self.st_name).spread
        self.spread_min = self.getdatabyname(self.st_name).spread_min
        self.open = self.getdatabyname(self.st_name).open
        self.high = self.getdatabyname(self.st_name).high
        self.low = self.getdatabyname(self.st_name).low
        self.close = self.getdatabyname(self.st_name).close
        self.volume = self.getdatabyname(self.st_name).volume  # leg1 volume
        self.vol_leg2 = self.getdatabyname(self.st_name).vol_leg2
        self.out_leg1 = self.getdatabyname(self.st_name).out_leg1
        self.out_leg2 = self.getdatabyname(self.st_name).out_leg2
        self.ytm_leg1 = self.getdatabyname(self.st_name).ytm_leg1
        self.ytm_leg2 = self.getdatabyname(self.st_name).ytm_leg2
        self.lend_rate_leg1 = self.getdatabyname(self.st_name).lend_rate_leg1
        self.lend_rate_leg2 = self.getdatabyname(self.st_name).lend_rate_leg2
        self.data = self.getdatabyname(self.st_name)

    def next(self):
        # calculate the lend fee
        if self.broker.getposition(data=self.data).size < 0:
            self.lend_fee += -self.data.lend_rate_leg1[-1] * self.broker.getposition(self.data).size / 365 * (
                    self.data.datetime.date(0) - self.data.datetime.date(-1)).days
        elif self.broker.getposition(data=self.data).size > 0:
            self.lend_fee += self.data.lend_rate_leg2[-1] * self.broker.getposition(self.data).size / 365 * (
                    self.data.datetime.date(0) - self.data.datetime.date(-1)).days
        # trading logic
        condition1 = (self.spread[0] >= -0.03)
        condition2 = (self.vol_leg2[0] < self.volume[0])
        condition3 = (self.spread[0] > self.spread_min[0] * 0.9)
        condition4 = (self.broker.getposition(data=self.getdatabyname(self.st_name)).size < 0)
        condition5 = (self.spread[0] >= -0.005)
        condition6 = (self.out_leg2[0] > self.out_leg1[0] * 0.8)
        condition7 = (self.out_leg2[0] < self.out_leg1[0] * 0.6)
        condition8 = (self.spread[0] <= -0.04)
        condition9 = (self.vol_leg2[0] - self.volume[0] > 100000000000)
        if condition7:
            if (not self.position) & condition1 & condition2:
                self.sell(size=5000000)
        if condition4 & condition3 & condition9 & condition8:
            self.close()
            self.buy(size=5000000)
        elif (not self.position) & condition3 & condition9 & condition8:
            self.buy(size=5000000)
        elif condition5 & condition6:
            self.close()
        # self.log(self.lend_fee)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'Order {order.ref}, BUY EXECUTED, {order.executed.price:.2f}, {order.executed.size:.2f}')
                self.log(f'Order {order.ref}, LEND FEE, {self.lend_fee:.2f}')
            elif order.issell():
                self.log(f'Order {order.ref}, SELL EXECUTED, {order.executed.price:.2f}, {order.executed.size:.2f}')
                if self.broker.getposition(self.data).size == 0:
                    self.log(
                        f'Order {order.ref}, LEND FEE, {self.lend_fee:.2f},CASH, {self.broker.get_cash():.2f},PROFIT,{self.broker.get_cash() - self.lend_fee - 50000000}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'Order {order.ref} Canceled/Margin/Rejected/Expired')

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"Trade Profit, Gross:{trade.pnl:.4f}, Net:{trade.pnlcomm:.4f}, Commission:{trade.commission:.4f}")


def main():
    # Create a cerebro entity
    for i in range(0, 13):
        leg1_code = SPREAD.CDB_CODES[i]
        leg2_code = SPREAD.CDB_CODES[i + 1]
        cerebro = bt.Cerebro()
        input_file = PATH.SPREAD_DATA + leg1_code + '_' + leg2_code + '_bt.csv'
        price_df = pd.read_csv(input_file, parse_dates=['DATE'])
        # minimum spread in the past 15 days are used as the threshold to open a position
        price_df['SPREAD_MIN'] = price_df['SPREAD'].rolling(15).min()
        price_df.loc[:, ['SPREAD_MIN']] = price_df.loc[:, ['SPREAD_MIN']].fillna(method='backfill')
        price_df = price_df.dropna()
        data1 = SpreadData(dataname=price_df, nocase=True)
        cerebro.adddata(data1, name=SpreadBaselineStrategy.st_name)
        cerebro.addstrategy(SpreadBaselineStrategy)
        cerebro.broker.set_cash(50000000.0)
        print(leg1_code + '_' + leg2_code)
        cerebro.run()
    # i=11
    # leg1_code = SPREAD.CDB_CODES[i]
    # leg2_code = SPREAD.CDB_CODES[i + 1]
    # cerebro = bt.Cerebro()
    # input_file = PATH.SPREAD_DATA + leg1_code + '_' + leg2_code + '_bt.csv'
    # price_df = pd.read_csv(input_file, parse_dates=['DATE'])
    # data1 = TBondData(dataname=price_df, nocase=True)
    # cerebro.adddata(data1, name=SpreadTrade.spread_name)
    # cerebro.addstrategy(SpreadTrade)
    # cerebro.broker.set_cash(50000000.0)
    # print(leg1_code + '_' + leg2_code)
    # cerebro.run()


if __name__ == '__main__':
    main()
