import pandas as pd
import datetime
import numpy as np
from fxincome.const import COUPON_TYPE
from fxincome.const import CASHFLOW_TYPE
from fxincome.const import ACCOUNT_TYPE
from dateutil.relativedelta import relativedelta

from fxincome.asset import Bond

"""
Position_Bond，处理交易，逐日盯市，计算和记录损益、DV01、Duration
每买入一次债券，都会产生一个新的Position_Bond，即使买的是同一只债券。
核心函数为move_ytm()

    Fields:
    __date(datetime): 跟踪position的日期，逐日盯市。position初始化时，__date为初始化日的前一日。
    account_type(Enum)：三大会计账户之一，
        ACCOUNT_TYPE.TPL
        ACCOUNT_TYPE.OCI
        ACCOUNT_TYPE.AC
    gain(DataFrame): 逐日盯市的损益、DV01、Duration等的计算结果
        date(datetime)， 盯市日
        quantity(float): 持有Position对应债券的数量，单位由外部用户决定
        market_cleanprice(float):债券在盯市日的估值净价，单位为/百元，形如100.01
        market_dirtyprice(float):债券在盯市日的估值全价，单位为/百元，形如101.05
        market_ytm(float)：债券在盯市日的ytm，形如3.5
        cost_cleanprice(float)：债券在盯市日的成本净价，单位为/百元，形如99.05
        coupon(float)：债券在盯市日当日的票面利息收入，和quantity有关，形如0.21
        interest(float)：债券在盯市日当日的利息收入，和quantity有关，形如0.21。
                        专为OCI和AC而设，=实际利率*折溢摊净价
                        对于TPL来说，interest = coupon
        price_gain(float)：债券在盯市日当日的价差收入，正负数皆有可能。只有卖出日才会产生price_gain。
        coupon_sum(float)：债券在买入日至盯市日的票面利息收入，和quantity有关，形如3.1
        interest_sum(float)：债券在买入日至盯市日的利息收入，和quantity有关，形如3.1。
                            专为OCI和AC而设，对于TPL，interest_sum = coupon_sum
        price_gain_sum(float)：债券在买入日至盯市日的价差收入，正负数皆有可能。
        float_gain_sum(float)：债券在盯市日的浮动盈亏
        gain_sum(float)：债券在买入日至盯市日的总损益， = interest_sum + price_gain_sum + float_gain_sum
        损益的单位和quantity的单位一致
        
"""

class Position_Bond:
    def __init__(self, pid, bond, account_type, begin_quantity, begin_date, begin_cleanprice):
        self.pid = pid
        self.bond = bond
        self.account_type = account_type
        self.begin_quantity = begin_quantity
        self.begin_date = begin_date
        self.begin_cleanprice = begin_cleanprice
        self.begin_dirtyprice = bond.cleanprice_to_dirtyprice(begin_date, begin_cleanprice)
        self.begin_ytm = bond.cleanprice_to_ytm(begin_date, begin_cleanprice)

        self.quantity = begin_quantity
        self.real_daily_rate = bond.amortprice_to_dailyrate(begin_date, begin_cleanprice)

        self.cashflow_history_df = pd.DataFrame([[begin_date, -begin_quantity * self.begin_dirtyprice / 100]],
                                                columns=['date', 'cash'])
        self.gain = pd.DataFrame([], columns=['date', 'quantity', 'market_cleanprice', 'market_dirtyprice',
                                              'market_ytm', 'cost_cleanprice', 'coupon', 'interest',
                                              'price_gain', 'coupon_sum', 'interest_sum', 'price_gain_sum',
                                              'float_gain_sum', 'gain_sum', 'dv01', 'duration'])
        self.__date = begin_date + datetime.timedelta(days=-1)

    """
    求该Position_Bond在self.__date前后的现金流
        Args:
            cashflow_type(Enum): 
                Undelivered: 返回self.__date（不含）之后的现金流
                History: 返回self.__date（含）之前的现金流
                All: 返回所有现金流，无视日期
        Returns:
            cashflow_df(DataFrame)：现金流, columns = ['date', 'cash', 'type']
    """

    def get_cashflow(self, cashflow_type):
        if cashflow_type == CASHFLOW_TYPE.Undelivered:
            cashflow_undelivered_df = self.bond.get_cashflow(self.__date, 'Undelivered')
            cashflow_undelivered_df['cash'] = cashflow_undelivered_df['cash'] * self.quantity / 100
            cashflow_undelivered_df['type'] = 'Undelivered'
            cashflow_undelivered_df = cashflow_undelivered_df[cashflow_undelivered_df['cash'] != 0].copy()
            return cashflow_undelivered_df
        elif cashflow_type == CASHFLOW_TYPE.History:
            cashflow_history_df = self.cashflow_history_df
            cashflow_history_df['type'] = 'History'
            return cashflow_history_df
        elif cashflow_type == CASHFLOW_TYPE.All:
            cashflow_undelivered_df = self.bond.get_cashflow(self.__date, 'Undelivered')
            cashflow_undelivered_df['cash'] = cashflow_undelivered_df['cash'] * self.quantity / 100
            cashflow_history_df = self.cashflow_history_df
            cashflow_undelivered_df['type'] = 'Undelivered'
            cashflow_undelivered_df = cashflow_undelivered_df[cashflow_undelivered_df['cash'] != 0].copy()
            cashflow_history_df['type'] = 'History'

            cashflow_all = pd.concat([cashflow_history_df, cashflow_undelivered_df])
            return cashflow_all
        else:
            raise NotImplementedError("Unknown CASHFLOW_TYPE")

    def get_position_gain(self):
        position_gain_df = self.gain
        return position_gain_df

    def move_curve(self, newdate, curve_df=None, quantity_delta=None):
        ytm = self.bond.curve_to_ytm(newdate, curve_df) if isinstance(curve_df, pd.core.frame.DataFrame) else None
        self.move_ytm(newdate, ytm, quantity_delta)

    """
    逐日盯市，处理交易，计算和记录损益、DV01、Duration
    每个Position_Bond只能向未来日期方向move_ytm，不能回溯，不能重复。
        原因是该函数使用内部的__date确保整个生命周期的完整性，每次move都会修改__date。
    每买入一次债券，都会产生一个新的Position_Bond，即使买的是同一只债券。
        Args:
            newdate(datetime): 盯市日，函数内部将验证 new_date > __date 
            ytm(float): 债券的市场估值ytm
            quantity_delta(float): 卖出债券的数量，不能>0。买入债券应新生成一个Position_Bond
    """

    def move_ytm(self, newdate, ytm=None, quantity_delta=None):
        if self.__date >= newdate:
            raise ValueError(f"move_ytm() cannot move backward. (id: {str(self.pid)} )")
        if self.quantity < 0:
            return
        if self.quantity == 0:
            last_df = self.gain.iloc[-1:, ].copy()
            if last_df.date.iat[0] != newdate:
                last_df.date.iat[0] = newdate
                last_df.loc[last_df.index[0], ['market_cleanprice', 'market_dirtyprice']] = 0
                last_df.market_ytm.iat[0] = np.nan
                self.gain = self.gain.append([last_df], ignore_index=True, sort=False)
                self.__date = newdate

        if self.quantity > 0:
            while (self.__date < newdate) and (self.__date < self.bond.end_date):
                self.__date += datetime.timedelta(days=1)
                # 算现金流
                if (self.__date in list(self.bond._cashflow_df['date'])) and (not self.gain.empty):
                    cash = self.bond._cashflow_df[self.bond._cashflow_df['date'] == self.__date].cash.iat[0] # 债券有新的现金流
                    cash = cash * self.gain.quantity.iat[-1] / 100  ## 乘以最新的quantity
                    self.cashflow_history_df = self.cashflow_history_df.append([{'date': self.__date, 'cash': cash}],
                                                                               ignore_index=True, sort=False)
                # 算收益
                coupon = self.bond.get_dailycoupon(self.__date) * self.quantity / 100

                if self.account_type == ACCOUNT_TYPE.TPL:
                    self.gain = self.gain.append([{'date': self.__date,
                                                   'quantity': self.quantity,
                                                   'coupon': coupon,
                                                   'interest': coupon,
                                                   'cost_cleanprice': self.begin_cleanprice}],
                                                    ignore_index=True, sort=False)

                elif self.account_type in [ACCOUNT_TYPE.OCI, ACCOUNT_TYPE.AC]:

                    if self.gain.empty:
                        cost_cleanprice=self.begin_cleanprice
                    else:
                        cost_cleanprice=self.gain.cost_cleanprice.iat[-1]* (1 + self.real_daily_rate / 100) - self.bond.get_dailycoupon(
                            self.__date+datetime.timedelta(days=-1))
                    interest = cost_cleanprice * self.real_daily_rate / 100 * self.quantity / 100 if coupon != 0 else 0

                    self.gain = self.gain.append([{'date': self.__date,
                                                   'quantity': self.quantity,
                                                   'coupon': coupon,
                                                   'interest': interest,
                                                   'cost_cleanprice': cost_cleanprice}],
                                                   ignore_index=True, sort=False)
                    pass
                else:
                    raise NotImplementedError("Unknown ACCOUNT_TYPE")
            # 债券到期，冲减position
            if self.__date == self.bond.end_date:
                ytm = np.nan
                quantity_delta = -self.quantity
            if ytm:
                self.gain.market_cleanprice.iat[-1] = self.bond.ytm_to_cleanprice(self.__date, ytm)
                self.gain.market_dirtyprice.iat[-1] = self.bond.ytm_to_dirtyprice(self.__date, ytm)
                self.gain.market_ytm.iat[-1] = ytm
                self.gain.float_gain_sum.iat[-1] = (self.gain.market_cleanprice.iat[-1] -
                                                    self.gain.cost_cleanprice.iat[-1]) * self.quantity / 100

                self.gain.dv01.iat[-1] = self.bond.ytm_to_dv01(self.__date, ytm) * self.quantity / 100
                self.gain.duration.iat[-1] = self.bond.ytm_to_duration(self.__date, ytm, 'Modified')

            # 卖出债券
            if quantity_delta:
                old_quantity = self.quantity
                self.quantity += quantity_delta
                if quantity_delta > 0:
                    raise ValueError(f"Should Open a New Position (id: {str(self.pid)}")
                if self.quantity < 0:
                    raise ValueError(f"Not Enough Bond to Sell (id: {str(self.pid)}")
                if self.__date == self.cashflow_history_df.iloc[-1, 0]:
                    self.cashflow_history_df.iloc[-1, 1] += -self.gain.market_dirtyprice.iat[-1]\
                                                            * quantity_delta / 100 \
                                                            * (self.__date != self.bond._cashflow_df.iloc[-1, 0])
                else:
                    self.cashflow_history_df = self.cashflow_history_df.append(
                        [{'date': self.__date, 'cash': -self.gain.iloc[-1, 3] * quantity_delta / 100}],
                        ignore_index=True, sort=False)
                self.gain.quantity.iat[-1] = self.quantity
                self.gain.coupon.iat[-1] = self.gain.coupon.iat[-1] / old_quantity * self.quantity
                self.gain.interest.iat[-1] = self.gain.interest.iat[-1] / old_quantity * self.quantity
                # 由于当日卖出债券，计算价差时，卖出部分的债券的cost_cleanprice应取上一日
                self.gain.price_gain.iat[-1] = (self.gain.market_cleanprice.iat[-1]
                                                - self.gain.cost_cleanprice.iat[-1]) \
                                               * (-quantity_delta) / 100
                self.gain.float_gain_sum.iat[-1] = (self.gain.market_cleanprice.iat[-1]
                                                    - self.gain.cost_cleanprice.iat[-1]) \
                                                   * self.quantity / 100
                self.gain.dv01.iat[-1] = self.bond.ytm_to_dv01(self.__date, ytm) * self.quantity / 100
                # self.gain.iloc[-1, 14] =0
                # self.gain.iloc[-1, 15] = 0

            self.gain['coupon_sum'] = self.gain['coupon'].cumsum()
            self.gain['interest_sum'] = self.gain['interest'].cumsum()
            self.gain['price_gain_sum'] = self.gain['price_gain'].fillna(0).cumsum()
            self.gain['gain_sum'] = self.gain['interest_sum'] + self.gain[
                'price_gain_sum'] + self.gain['float_gain_sum']


if __name__ == '__main__':
    code = '200016'
    initial_date = datetime.datetime(2020, 11, 19)
    end_date = datetime.datetime(2030, 11, 19)
    issue_price = 100
    coupon_rate = 3.27
    coupon_type = '附息'
    coupon_frequency = 2
    a = Bond(code, initial_date, end_date, issue_price, coupon_rate, coupon_type, coupon_frequency)
