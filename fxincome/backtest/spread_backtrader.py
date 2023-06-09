import backtrader as bt
import pandas as pd
from fxincome.const import PATH, SPREAD


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
        self.spread_data = self.getdatabyname(self.ST_NAME)
        self.spread = self.spread_data.spread
        self.spread_min = self.spread_data.spread_min
        self.open = self.spread_data.open
        self.high = self.spread_data.high
        self.low = self.spread_data.low
        self.volume = self.spread_data.volume  # leg1 volume
        self.vol_leg2 = self.spread_data.vol_leg2
        self.out_leg1 = self.spread_data.out_leg1
        self.out_leg2 = self.spread_data.out_leg2
        self.ytm_leg1 = self.spread_data.ytm_leg1
        self.ytm_leg2 = self.spread_data.ytm_leg2
        self.lend_rate_leg1 = self.spread_data.lend_rate_leg1
        self.lend_rate_leg2 = self.spread_data.lend_rate_leg2
        self.lend_fee = 0.0
        self.total_fee = 0.0

    def next(self):
        if self.getposition(self.spread_data).size < 0:
            self.lend_fee = self.__calculate_lend_fee(size=self.getposition(self.spread_data).size,
                                                      rate=self.lend_rate_leg1[0], type='borrow')
        elif self.getposition(self.spread_data).size > 0:
            self.lend_fee = self.__calculate_lend_fee(size=self.getposition(self.spread_data).size,
                                                      rate=self.lend_rate_leg2[0], type='borrow')
        self.broker.add_cash(self.lend_fee)

        # trading logic
        condition1 = (self.spread[0] >= -0.03)
        condition2 = (self.vol_leg2[0] < self.volume[0])
        condition3 = (self.spread[0] > self.spread_min[0] * 0.9)
        condition4 = (self.getposition(self.spread_data).size < 0)
        condition5 = (self.spread[0] >= -0.005)
        condition6 = (self.out_leg2[0] > self.out_leg1[0] * 0.8)
        condition7 = (self.out_leg2[0] < self.out_leg1[0] * 0.6)
        condition8 = (self.spread[0] <= -0.04)
        condition9 = (self.vol_leg2[0] - self.volume[0] > 1e11)
        condition10 = (self.getposition(self.spread_data).size == 0)
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
                if self.broker.getposition(self.spread_data).size == 0:
                    self.log(
                        f'Order: {order.ref},'
                        f'CASH: {self.broker.get_cash():.2f},'
                        f'PROFIT: {self.broker.get_cash() - self.INIT_CASH:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'Order {order.ref} Canceled/Margin/Rejected/Expired')

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"Trade Profit, Gross:{trade.pnl:.2f}, Net:{trade.pnlcomm:.2f}, Commission:{trade.commission:.2f}")

    def __calculate_lend_fee(self, size: int, rate: float, type: str):
        fee = 0
        if type == 'borrow':
            fee = -rate * abs(size) / 365 * (self.spread_data.datetime.date(1) - self.spread_data.datetime.date(0)).days
        elif type == 'lend':
            fee = rate * abs(size) / 365 * (self.spread_data.datetime.date(1) - self.spread_data.datetime.date(0)).days
        return fee


def main():
    # for i in range(0, 13):
    i = 0
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
    print(leg1_code + '_' + leg2_code)
    cerebro.run()


if __name__ == '__main__':
    main()
