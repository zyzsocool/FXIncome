import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fxincome.const import PATH, SPREAD
from fxincome import logger


class SpreadData(bt.feeds.PandasData):
    # Lines besides those required by bactrader (datetime, OHLC, volume, openinterest).
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
    ST_NAME = 'baseline'
    SIZE = 1000000  # 1 million size for 100 million face value of bond
    INIT_CASH = 1e7  # 10 million initial cash for 100 million face value of bond

    def log(self, txt, dt=None):
        """ Logging function for this strategy"""
        dt = dt or self.getdatabyname(self.ST_NAME).datetime.date(0)
        print(f"{dt:%Y%m%d} - {txt}")

    def __init__(self):
        self.data = self.getdatabyname(self.ST_NAME)
        self.spread = self.data.spread
        self.spread_min = self.data.spread_min
        self.open = self.data.open
        self.high = self.data.high
        self.low = self.data.low
        self.volume = self.data.volume  # leg1 volume
        self.vol_leg2 = self.data.vol_leg2
        self.out_leg1 = self.data.out_leg1
        self.out_leg2 = self.data.out_leg2
        self.ytm_leg1 = self.data.ytm_leg1
        self.ytm_leg2 = self.data.ytm_leg2
        self.lend_rate_leg1 = self.data.lend_rate_leg1
        self.lend_rate_leg2 = self.data.lend_rate_leg2
        self.total_fee = 0.0
        self.leg1_code = int(self.data.code_leg1[0])
        self.cash_flow_leg1 = pd.read_csv(PATH.SPREAD_DATA + f'cash_flow_{self.leg1_code}.csv', parse_dates=['DATE'])
        self.cash_flow_leg1['DATE'] = self.cash_flow_leg1['DATE'].dt.date
        self.leg2_code = int(self.data.code_leg2[0])
        self.cash_flow_leg2 = pd.read_csv(PATH.SPREAD_DATA + f'cash_flow_{self.leg2_code}.csv', parse_dates=['DATE'])
        self.cash_flow_leg2['DATE'] = self.cash_flow_leg2['DATE'].dt.date
        self.result = pd.read_csv(PATH.SPREAD_DATA + f'{self.leg1_code}_{self.leg2_code}_bt.csv', parse_dates=['DATE'])
        self.result['DATE'] = self.result['DATE'].dt.date
        self.result['Profit'] = 0.0
        self.result['Lend Fee'] = 0.0
        self.result['Sell'] = 0.0
        self.result['Buy'] = 0.0
        self.last_leg1_remaining_payment_times = len(
            self.cash_flow_leg1[self.cash_flow_leg1['DATE'] > self.data.datetime.date(1)])
        self.last_leg2_remaining_payment_times = len(
            self.cash_flow_leg2[self.cash_flow_leg2['DATE'] > self.data.datetime.date(1)])
        self.leg1_the_ordinal_of_next_payment = len(self.cash_flow_leg1) - self.last_leg1_remaining_payment_times
        self.leg2_the_ordinal_of_next_payment = len(self.cash_flow_leg2) - self.last_leg2_remaining_payment_times

    def next(self):
        lend_fee = 0.0
        #  Only fee for borrowing is considered.
        if self.getposition(self.data).size < 0:
            lend_fee = self.__calculate_lend_fee(face_value=self.getposition(self.data).size * 100,
                                                 rate=self.lend_rate_leg1[0], direction='borrow')
        elif self.getposition(self.data).size > 0:
            lend_fee = self.__calculate_lend_fee(face_value=self.getposition(self.data).size * 100,
                                                 rate=self.lend_rate_leg2[0], direction='borrow')
        self.broker.add_cash(lend_fee)
        self.total_fee += lend_fee
        self.result.loc[self.result['DATE'] == self.data.datetime.date(0), 'Lend Fee'] = self.total_fee

        # coupon payment
        self.today_leg1_remaining_payment_times = len(
            self.cash_flow_leg1[self.cash_flow_leg1['DATE'] > self.data.datetime.date(0)])
        self.today_leg2_remaining_payment_times = len(
            self.cash_flow_leg2[self.cash_flow_leg2['DATE'] > self.data.datetime.date(0)])
        self.leg1_the_ordinal_of_next_payment = len(self.cash_flow_leg1) - self.last_leg1_remaining_payment_times
        self.leg2_the_ordinal_of_next_payment = len(self.cash_flow_leg2) - self.last_leg2_remaining_payment_times
        if self.today_leg1_remaining_payment_times < self.last_leg1_remaining_payment_times:
            coupon = self.cash_flow_leg1.loc[[self.leg1_the_ordinal_of_next_payment], ['AMOUNT']].values[0][
                         0] * self.last_position
            self.log(f'coupon payment {coupon}')
            self.broker.add_cash(coupon)
        if self.today_leg2_remaining_payment_times < self.last_leg2_remaining_payment_times:
            coupon = -self.cash_flow_leg2.loc[[self.leg2_the_ordinal_of_next_payment], ['AMOUNT']].values[0][
                0] * self.last_position
            self.log(f'coupon payment {coupon}')
            self.broker.add_cash(coupon)
        self.last_leg1_remaining_payment_times = self.today_leg1_remaining_payment_times
        self.last_leg2_remaining_payment_times = self.today_leg2_remaining_payment_times
        self.last_position = self.getposition(self.data).size

        # trading logic
        condition1 = (self.spread[0] >= -0.03)
        condition2 = (self.vol_leg2[0] < self.volume[0])
        condition3 = (self.spread[0] > self.spread_min[0] * 0.9)
        condition4 = (self.getposition(self.data).size < 0)
        condition5 = (self.spread[0] >= -0.005)
        condition6 = (self.out_leg2[0] > self.out_leg1[0] * 0.8)
        condition7 = (self.out_leg2[0] < self.out_leg1[0] * 0.6)
        condition8 = (self.spread[0] <= -0.04)
        condition9 = (self.vol_leg2[0] - self.volume[0] > 1e11)
        condition10 = (self.getposition(self.data).size == 0)
        if condition10 & condition1 & condition2 & condition7:
            self.sell(size=self.SIZE)
        if condition3 & condition8 & condition9:
            if condition4:
                self.close()
                self.buy(size=self.SIZE)
            elif condition10:
                self.buy(size=self.SIZE)
        elif condition5 & condition6:
            self.close()
        self.result.loc[
            self.result['DATE'] == self.data.datetime.date(0), 'Profit'] = self.broker.getvalue() - self.INIT_CASH
        # if self.getposition(self.data).size == 0:
        #     self.result.to_csv(PATH.SPREAD_DATA + f'{self.leg1_code}_{self.leg2_code}_result.csv', index=False)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'Order {order.ref}, BUY EXECUTED, {order.executed.price:.2f}, {order.executed.size:.2f}')
                self.result.loc[self.result['DATE'] == self.data.datetime.date(0), 'Buy'] = 1
            elif order.issell():
                self.log(f'Order {order.ref}, SELL EXECUTED, {order.executed.price:.2f}, {order.executed.size:.2f}')
                self.result.loc[self.result['DATE'] == self.data.datetime.date(0), 'Sell'] = 1
                if self.broker.getposition(self.data).size == 0:
                    self.log(
                        f'Order: {order.ref},'
                        f'CASH: {self.broker.get_cash():.2f},'
                        f'FEE: {self.total_fee:.2f},'
                        f'PROFIT: {self.broker.get_cash() - self.INIT_CASH:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'Order {order.ref} Canceled/Margin/Rejected/Expired')

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"Trade Profit, Gross:{trade.pnl:.2f}, Net:{trade.pnlcomm:.2f}, Commission:{trade.commission:.2f}")

    def __calculate_lend_fee(self, face_value: int, rate: float, direction: str) -> float:
        """
        Calculate the lending fee for a bond. Direction must be 'borrow' or 'lend'.
        Args:
            face_value (int):   face value of the bond
            rate (float):       lending rate. 1 means 1% annual rate.
            direction (str):    'borrow' means the bond is borrowed and must pay lending fee.
                                'lend' means the bond is lent out and receive lending fee.

        Returns:
            fee(float): lending fee. Positive means receive fee, negative means pay fee.
        """
        rate = rate / 100
        if direction == 'borrow':
            fee = -rate * abs(face_value) / 365 * (self.data.datetime.date(1) - self.data.datetime.date(0)).days
        elif direction == 'lend':
            fee = rate * abs(face_value) / 365 * (self.data.datetime.date(1) - self.data.datetime.date(0)).days
        else:
            raise ValueError("direction must be 'borrow' or 'lend'")
        return fee


def plot_spread(leg1, leg2):
    input_file = PATH.SPREAD_DATA + f'{leg1}_{leg2}_result.csv'
    data = pd.read_csv(input_file, parse_dates=['DATE'])
    data['Profit'] = data['Profit'].apply(lambda x: x if x > -2000000 else np.nan)
    data['Profit'] = data['Profit'].fillna(method='ffill')
    fig = plt.figure(num=1, figsize=(15, 5))
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    lns1 = ax.plot(data['DATE'], data['SPREAD'], color='r', label='SPREAD')
    lns = lns1
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread")
    for i in range(0, len(data['Buy'])):
        if data['Buy'][i] == 1:
            ax.annotate('Buy', xy=(data['DATE'][i], data['SPREAD'][i]),
                        xytext=(data['DATE'][i], data['SPREAD'][i] + 0.01),
                        arrowprops=dict(facecolor='green', shrink=0.05), )
    for i in range(0, len(data['Sell'])):
        if data['Sell'][i] == 1:
            ax.annotate('Sell', xy=(data['DATE'][i], data['SPREAD'][i]),
                        xytext=(data['DATE'][i], data['SPREAD'][i] + 0.01),
                        arrowprops=dict(facecolor='yellow', shrink=0.05), )
    lns2 = ax2.plot(data['DATE'], data['Profit'], label='Profit')
    lns += lns2
    ax2.set_ylabel("Profit")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=10)
    plt.title(f'{leg1} and {leg2} Spread Trading')
    plt.tight_layout()
    plt.show()


def main():
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
        cerebro.adddata(data1, name=SpreadBaselineStrategy.ST_NAME)
        cerebro.addstrategy(SpreadBaselineStrategy)
        cerebro.broker.set_cash(SpreadBaselineStrategy.INIT_CASH)
        logger.info(leg1_code + '_' + leg2_code)
        cerebro.run()
        logger.info(f'PROFIT: {(cerebro.broker.get_cash() - SpreadBaselineStrategy.INIT_CASH) / 10000:.2f}')
    # for i in range(0, 13):
    #     leg1_code = SPREAD.CDB_CODES[i]
    #     leg2_code = SPREAD.CDB_CODES[i + 1]
    #     plot_spread(leg1_code,leg2_code)


if __name__ == '__main__':
    main()
