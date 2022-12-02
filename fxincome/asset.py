import pandas as pd
from fxincome.const import COUPON_TYPE
from fxincome.const import CASHFLOW_TYPE
import datetime
import calendar
import numpy as np
from dateutil.relativedelta import relativedelta
from scipy import optimize


def f_npv(ytm, time, cash):
    npv = 0
    for t, c in zip(time, cash):
        npv += c / ((1 + ytm) ** t)
    return npv


def get_curve(points, type):
    """
    使用收益率曲线中已知的一组点，通过指定的拟合算法生成任意点的收益率。
    该函数返回一个收益率曲线函数func(x)，使用者利用func(x)即可计算期限对应的收益率。
    对于一个point(x, y)，x坐标是年（单位为年），y坐标是收益率（单位为%）。
        Args:
            points(ndarray): 收益率曲线中已知的点，形状是2D Array [[x1,y1], [x2,y2] ... [xn,yn]]，其中x不能重复
            type(str): [LINEAR, POLYNOMIAL, HERMIT, SPLINE]
        Returns:
            func(float): 输入任意年限，输出对应的拟合收益率。输入单位是年，输出单位是%
    """

    size = points.shape[0]
    if type == 'LINEAR':
        points = points[points[:, 0].argsort()]  # 以x坐标（年限）作升序排序

        def func(x):
            for i in range(1, size):
                if x <= points[i, 0]:
                    break
            return (points[i, 1] - points[i - 1, 1]) / (points[i, 0] - points[i - 1, 0]) * (x - points[i - 1, 0]) + \
                   points[
                       i - 1, 1]
    elif type == 'POLYNOMIAL':
        matrix_x = np.zeros([size, size])
        matrix_y = np.array(points[:, 1])
        for i in range(size):
            for j in range(size):
                matrix_x[i, j] = points[i, 0] ** j
        para = np.dot(np.linalg.inv(matrix_x), matrix_y)

        def func(x):
            xx = np.array([x ** i for i in range(size)])
            return np.dot(para, xx)
    elif type == 'HERMIT':
        matrix_x = np.zeros([(size - 1) * 4, (size - 1) * 4])
        matrix_y = np.zeros([(size - 1) * 4])
        y_1 = [(points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])] + \
              [(points[i + 1, 1] - points[i - 1, 1]) / (points[i + 1, 0] - points[i - 1, 0]) for i in
               range(1, size - 1)] + \
              [(points[size - 1, 1] - points[size - 2, 1]) / (points[size - 1, 0] - points[size - 2, 0])]
        for i in range(size - 1):
            for j in range(2):
                matrix_x[2 * i + j, 4 * i] = points[i + j, 0] ** 3
                matrix_x[2 * i + j, 4 * i + 1] = points[i + j, 0] ** 2
                matrix_x[2 * i + j, 4 * i + 2] = points[i + j, 0]
                matrix_x[2 * i + j, 4 * i + 3] = 1
                matrix_y[2 * i + j] = points[i + j, 1]

                matrix_x[2 * (size - 1) + 2 * i + j, 4 * i] = 3 * points[i + j, 0] ** 2
                matrix_x[2 * (size - 1) + 2 * i + j, 4 * i + 1] = 2 * points[i + j, 0]
                matrix_x[2 * (size - 1) + 2 * i + j, 4 * i + 2] = 1
                matrix_y[2 * (size - 1) + 2 * i + j] = y_1[i + j]
        para = np.dot(np.linalg.inv(matrix_x), matrix_y)

        def func(x):
            xx = np.zeros((size - 1) * 4)
            for i in range(1, size):
                if x <= points[i, 0]:
                    break
            xx[4 * (i - 1)] = x ** 3
            xx[4 * (i - 1) + 1] = x ** 2
            xx[4 * (i - 1) + 2] = x
            xx[4 * (i - 1) + 3] = 1
            return np.dot(para, xx)
    elif type == 'SPLINE':
        matrix_x = np.zeros([(size - 1) * 4, (size - 1) * 4])
        matrix_y = np.zeros([(size - 1) * 4])
        for i in range(size - 1):
            for j in range(2):
                matrix_x[2 * i + j, 4 * i] = points[i + j, 0] ** 3
                matrix_x[2 * i + j, 4 * i + 1] = points[i + j, 0] ** 2
                matrix_x[2 * i + j, 4 * i + 2] = points[i + j, 0]
                matrix_x[2 * i + j, 4 * i + 3] = 1
                matrix_y[2 * i + j] = points[i + j, 1]
        for i in range(size - 2):
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i] = 3 * points[i + 1, 0] ** 2
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i + 1] = 2 * points[i + 1, 0]
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i + 2] = 1
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i + 4] = -3 * points[i + 1, 0] ** 2
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i + 5] = -2 * points[i + 1, 0]
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i + 6] = -1

            matrix_x[(size - 1) * 2 + 2 * i + 1, 4 * i] = 6 * points[i + 1, 0]
            matrix_x[(size - 1) * 2 + 2 * i + 1, 4 * i + 1] = 2
            matrix_x[(size - 1) * 2 + 2 * i + 1, 4 * i + 4] = -6 * points[i + 1, 0]
            matrix_x[(size - 1) * 2 + 2 * i + 1, 4 * i + 5] = -2
            matrix_x[(size - 1) * 2 + 2 * i + 1, 4 * i + 7] = -1
        matrix_x[-2, 0] = 6 * points[0, 0]
        matrix_x[-2, 1] = 2
        matrix_x[-1, -4] = 6 * points[-1, 0]
        matrix_x[-1, -3] = 2
        para = np.dot(np.linalg.inv(matrix_x), matrix_y)

        def func(x):
            xx = np.zeros((size - 1) * 4)
            for i in range(1, size):
                if x <= points[i, 0]:
                    break
            xx[4 * (i - 1)] = x ** 3
            xx[4 * (i - 1) + 1] = x ** 2
            xx[4 * (i - 1) + 2] = x
            xx[4 * (i - 1) + 3] = 1
            return np.dot(para, xx)
    else:
        raise NotImplementedError("Unknown fitting method")
    return func

class Bond:
    def __init__(self, code, initial_date, end_date, issue_price, coupon_rate, coupon_type, coupon_frequency,
                 coupon_period=None, bond_name=None, bond_type=None):
        self.code = code
        self.initial_date = initial_date
        self.end_date = end_date
        self.face_value = 100
        self.issue_price = issue_price  # 发行价格，附息债券都为100，这个字段主要为贴现债券
        self.coupon_rate = coupon_rate  # 每100元的利息
        self.coupon_type = coupon_type
        self.coupon_frequency = coupon_frequency
        self.coupon_period = coupon_period if coupon_period else end_date.year - initial_date.year  # 债券期限（年），如果是整年的债券就可以不输入coupon_period,非整年的就要输入
        self._cashflow_df = self.cal_cashflow()
        self.bond_name = bond_name
        self.bond_type = bond_type

    def get_dailycoupon(self, date):
        cashflow_df = self.get_cashflow(date, 'Undelivered_Lastone')
        if cashflow_df.shape[0] <= 1:
            if cashflow_df.iloc[0, 0] == date:  # 1刚好在到期日这天
                dailycoupon = 0
                return dailycoupon
            else:  # 2晚于到期日
                raise Exception('The bond is due' + '(' + self.code + ')')
        interval_days = (cashflow_df.iloc[1, 0] - cashflow_df.iloc[0, 0]).days
        if self.coupon_type == COUPON_TYPE.REGULAR:
            dailycoupon = self.coupon_rate / self.coupon_frequency / interval_days
        elif self.coupon_type == COUPON_TYPE.DUE:
            dailycoupon = self.coupon_rate / interval_days
        elif self.coupon_type == COUPON_TYPE.ZERO:
            dailycoupon = (self.face_value - self.issue_price) / interval_days
        return dailycoupon

    def cal_cashflow(self):
        cashflow_df = pd.DataFrame([], columns=['date', 'cash'])
        if self.coupon_type == COUPON_TYPE.REGULAR:
            coupon = self.coupon_rate / self.coupon_frequency
            coupon_months = int(12 / self.coupon_frequency)  # 每隔多少个月付息一次
            coupon_times = int(self.coupon_frequency * (self.coupon_period))  # 总付息次数
            cashflow_df = cashflow_df.append([{'date': self.initial_date, 'cash': -self.face_value}], ignore_index=True)
            for i in range(coupon_times - 1):
                date = self.initial_date + relativedelta(months=coupon_months * (i + 1))
                cashflow_df = cashflow_df.append([{'date': date, 'cash': coupon}], ignore_index=True)
            cashflow_df = cashflow_df.append([{'date': self.end_date, 'cash': self.face_value + coupon}],
                                             ignore_index=True)
        elif self.coupon_type == COUPON_TYPE.ZERO:
            cashflow_df = cashflow_df.append([{'date': self.initial_date, 'cash': -self.issue_price}],
                                             ignore_index=True)
            cashflow_df = cashflow_df.append([{'date': self.end_date, 'cash': self.face_value}], ignore_index=True)
        elif self.coupon_type == COUPON_TYPE.DUE:
            cashflow_df = cashflow_df.append([{'date': self.initial_date, 'cash': -self.face_value}], ignore_index=True)
            cashflow_df = cashflow_df.append([{'date': self.end_date, 'cash': self.coupon_rate + self.face_value}],
                                             ignore_index=True)
        else:
            raise NotImplementedError("Unknown COUPON_TYPE")
        return cashflow_df

    """
    求该Bond在指定日期前后的现金流
        Args:
            date(datetime): 指定日期
            cashflow_type(Enum): 
                Undelivered: 返回指定日期（不含）之后的现金流
                History: 返回指定日期（含）之前的现金流
                Undelivered_Lastone: 返回指定日期之后的现金流 + 指定日期之前最后一个现金流
                All: 返回所有现金流，无视指定日期
        Returns:
            cashflow_df(DataFrame)：现金流
    """

    def get_cashflow(self, date, cashflow_type='Undelivered'):
        cashflow_df = self._cashflow_df
        if cashflow_type == CASHFLOW_TYPE.Undelivered:
            cashflow_df = cashflow_df[cashflow_df['date'] > date].copy()
        elif cashflow_type == CASHFLOW_TYPE.History:
            cashflow_df = cashflow_df[cashflow_df['date'] <= date].copy()
        elif cashflow_type == CASHFLOW_TYPE.Undelivered_Lastone:
            begin_num = sum(cashflow_df['date'].apply(lambda x: x <= date)) - 1
            cashflow_df = cashflow_df.iloc[begin_num:, :].copy()
        elif cashflow_type == CASHFLOW_TYPE.All:
            cashflow_df = self._cashflow_df
        else:
            raise NotImplementedError("Unknown CASHFLOW_TYPE")
        return cashflow_df

    """
    求该Bond在指定日期、指定ytm的折现全价，算法采用央行规则
        Args:
            date(datetime): 指定日期
            ytm(float): 指定ytm，放大100倍，形如3.5
            full_info(Boolean): 是否返回折现全价现金流
        Returns:
            cashflow_df(DataFrame)：折现全价现金流
            dirtyprice(float): 折现全价现金流之和
    """

    def ytm_to_dirtyprice(self, date, ytm, full_info=False):
        cashflow_df = self.get_cashflow(date, 'Undelivered_Lastone')
        if cashflow_df.shape[0] <= 1:
            if cashflow_df.iat[0, 0] == date:  # 1刚好在到期日这天
                dirtyprice = self.issue_price
                return dirtyprice
            else:  # 2晚于到期日
                raise Exception('The bond is due' + '(' + self.code + ')')
        # 3到期日之前
        interval_days = (cashflow_df.iat[1, 0] - cashflow_df.iat[0, 0]).days  # 当前付息区间的天数
        t = (cashflow_df.iat[1, 0] - date).days  # 距下一个付息日的天数
        cashflow_df = cashflow_df.iloc[1:, :]
        if self.coupon_type == COUPON_TYPE.REGULAR:
            if cashflow_df.shape[0] > 1:
                cashflow_df['deflator'] = [1 / (1 + ytm / 100 / self.coupon_frequency) ** (t / interval_days + i) for i
                                           in range(cashflow_df.shape[0])]
            # 只剩下最后一个付息期
            elif cashflow_df.shape[0] == 1:
                year_days = 366 if calendar.isleap(cashflow_df['date'].iat[0].year) else 365  # 闰年366天，其余365天
                cashflow_df['deflator'] = [1 / (1 + ytm / 100 * t / year_days)]
        # 剩余期限不超过1年的贴现债券、剩余期限不超过1年的到期还本付息债券
        elif self.coupon_type in [COUPON_TYPE.ZERO, COUPON_TYPE.DUE]:
            year_days = 366 if calendar.isleap(cashflow_df['date'].iat[0].year) else 365  # 闰年366天，其余365天
            cashflow_df['deflator'] = [1 / (1 + ytm / 100 * t / year_days)]
        cashflow_df['cash_deflated'] = cashflow_df['cash'] * cashflow_df['deflator']
        dirtyprice = cashflow_df['cash_deflated'].sum()
        # cashflow_df可能是个有用的dataframe
        if full_info:
            return cashflow_df
        else:
            return dirtyprice

    """
    求该Bond在指定日期、指定全价的ytm，算法采用牛顿法和央行规则，从ytm=5.0开始尝试
        Args:
            date(datetime): 指定日期
            dirtyprice(float): 指定全价
        Returns:
            ytm(float): 到期收益率
    """

    def dirtyprice_to_ytm(self, date, dirtyprice):
        ytm = 5.0
        dirtyprice_cal = self.ytm_to_dirtyprice(date, ytm)
        while abs(dirtyprice_cal - dirtyprice) > 0.00001:
            k = (self.ytm_to_dirtyprice(date, ytm + 0.00005) - self.ytm_to_dirtyprice(date, ytm - 0.00005)) / 0.0001
            b = dirtyprice_cal - k * ytm
            ytm = (dirtyprice - b) / k
            dirtyprice_cal = self.ytm_to_dirtyprice(date, ytm)
        return ytm

    def accrued_interest(self, date):
        cashflow_df = self.get_cashflow(date, 'Undelivered_Lastone')
        if cashflow_df.shape[0] <= 1:
            if cashflow_df.shape[0] <= 1:
                if cashflow_df.iloc[0, 0] == date:  # 1刚好在到期日这天
                    accrued_interest = 0
                    return accrued_interest
                else:  # 2晚于到期日
                    raise Exception('The bond is due' + '(' + self.code + ')')
        interval_days = (cashflow_df.iloc[1, 0] - cashflow_df.iloc[0, 0]).days  # 当前付息区间的天数
        t = (date - cashflow_df.iloc[0, 0]).days  # 距上一个付息日的天数
        if self.coupon_type == COUPON_TYPE.REGULAR:
            accrued_interest = self.coupon_rate / self.coupon_frequency * t / interval_days
        elif self.coupon_type == COUPON_TYPE.DUE:
            accrued_interest = self.coupon_rate * t / interval_days
        elif self.coupon_type == COUPON_TYPE.ZERO:
            accrued_interest = (self.face_value - self.issue_price) * t / interval_days
        return accrued_interest

    def dirtyprice_to_cleanprice(self, date, dirtyprice):
        accrued_interest = self.accrued_interest(date)
        return dirtyprice - accrued_interest

    def cleanprice_to_dirtyprice(self, date, cleanprice):
        accrued_interest = self.accrued_interest(date)
        return cleanprice + accrued_interest

    def ytm_to_cleanprice(self, date, ytm):
        dirtyprice = self.ytm_to_dirtyprice(date, ytm)
        cleanprice = self.dirtyprice_to_cleanprice(date, dirtyprice)
        return cleanprice

    def cleanprice_to_ytm(self, date, cleanprice):
        dirtyprice = self.cleanprice_to_dirtyprice(date, cleanprice)
        ytm = self.dirtyprice_to_ytm(date, dirtyprice)
        return ytm

    """
    辅助函数，求出该Bond在指定日期的剩余期限，再用该剩余期限在收益率曲线上用'插值法'计算到期收益率
    例子：
    设收益率曲线为 [30d, 2.5], [90d, 2.8], 
    则剩余期限为87d的ytm = 2.5 + (2.8-2.5) * (87-30)/(90-30) 
        Args:
            date(datetime): 指定日期
            curve_df(DataFrame): 收益率曲线
        Returns:
            ytm(float)：Yield to maturity
    """

    def curve_to_ytm(self, date, curve_df):
        days = (self.end_date - date).days
        begin_num = sum(curve_df['days'].apply(lambda x: x <= days)) - 1
        curve_df = curve_df.iloc[begin_num:begin_num + 2, :]
        if days < 0:
            return None
        elif curve_df.iloc[-1, 0] == days:
            return curve_df.iloc[-1, 1]
        elif curve_df.shape[0] < 2:
            raise Exception('The curve is too short' + '(' + self.code + ')')
        else:
            ytm = (curve_df.iloc[1, 1] - curve_df.iloc[0, 1]) / (curve_df.iloc[1, 0] - curve_df.iloc[0, 0]) * (
                    days - curve_df.iloc[0, 0]) + curve_df.iloc[0, 1]
            return ytm

    def curve_to_dirtyprice(self, date, curve_df):
        ytm = self.curve_to_ytm(date, curve_df)
        dirtyprice = self.ytm_to_dirtyprice(date, ytm)
        return dirtyprice

    def curve_to_cleanprice(self, date, curve_df):
        ytm = self.curve_to_ytm(date, curve_df)
        cleanprice = self.ytm_to_cleanprice(date, ytm)
        return cleanprice

    """
    从折溢摊净价推算出实际日利率，用于OCI账户或AC账户的折溢摊。具体算法为牛顿法
    该实际日利率存储于Position中（构造函数中通过初始净价推算实际日利率），以便计算各日的折溢摊净价。
        Args:
            date(datetime): 指定日期
            amort_price(float): Amortized Price，折溢摊净价
        Returns:
            real_daily_rate(float)：实际日利率
    """

    def amortprice_to_dailyrate(self, date, amort_price):
        real_daily_rate = 5 / 365
        price = self.__dailyrate_to_amortprice(date, real_daily_rate)
        while abs(price - amort_price) > 0.000001:
            k = (self.__dailyrate_to_amortprice(date, real_daily_rate + 0.00000001) -
                 self.__dailyrate_to_amortprice(date, real_daily_rate - 0.00000001)) / 0.00000002
            b = price - k * real_daily_rate
            real_daily_rate = (amort_price - b) / k
            price = self.__dailyrate_to_amortprice(date, real_daily_rate)
        return real_daily_rate

    """
    辅助函数，用于辅助amortprice_to_dailyrate()函数，辅助计算OCI账户和AC账户中债券的实际日利率
        Args:
            date(datetime): 指定日期
            real_daily_rate(float): 会计上的实际日利率
        Returns:
            amortized_price(float)：折溢摊净价
    """

    def __dailyrate_to_amortprice(self, date, real_daily_rate):
        real_daily_rate = real_daily_rate / 100
        cashflow_date_df = list(self.get_cashflow(date, 'Undelivered_Lastone')['date'])
        if self.coupon_type == COUPON_TYPE.REGULAR:
            coupon = self.coupon_rate / self.coupon_frequency
        elif self.coupon_type == COUPON_TYPE.DUE:
            coupon = self.coupon_rate
        elif self.coupon_type == COUPON_TYPE.ZERO:
            coupon = (self.face_value - self.issue_price)
        else:
            raise NotImplementedError("Unknown COUPON_TYPE")
        Tlist = []
        tlist = []
        nlist = []
        n = 0
        for date_i in cashflow_date_df[1:]:
            Tlist.append((cashflow_date_df[-1] - date_i).days)
            tlist.append((date_i - cashflow_date_df[n]).days)
            nlist.append((date_i - cashflow_date_df[n]).days)
            n += 1
        nlist[0] = (cashflow_date_df[1] - date).days
        Tlist = np.array(Tlist)
        tlist = np.array(tlist)
        nlist = np.array(nlist)
        price_coupon = sum(
            (coupon / tlist * ((1 + real_daily_rate) ** nlist - 1) / real_daily_rate) * (1 + real_daily_rate) ** Tlist)
        amortized_price = (100 + price_coupon) / ((1 + real_daily_rate) ** sum(nlist))
        return amortized_price

    def ytm_to_dv01(self, date, ytm):
        if not ytm:
            return np.nan
        elif date < self.end_date:
            dirtyprice_up = self.ytm_to_dirtyprice(date, ytm + 0.005)
            dirtyprice_down = self.ytm_to_dirtyprice(date, ytm - 0.005)
            dv01 = (dirtyprice_up - dirtyprice_down)
            return dv01
        else:
            return 0

    def curve_to_dv01(self, date, curve_df):
        ytm = self.curve_to_ytm(date, curve_df)
        dv01 = self.ytm_to_dv01(date, ytm)
        return dv01

    """
    求出该Bond在指定日期、指定ytm的久期
        Args:
            date(datetime): 指定日期
            ytm(float): 到期收益率
            DURATION_TYPE(str): Macaulay or Modified
        Returns:
            duration(float)：Duration
    """

    def ytm_to_duration(self, date, ytm, DURARION_TYPE):
        if not ytm:
            return np.nan
        elif date < self.end_date:
            cashflow_df = self.ytm_to_dirtyprice(date, ytm, True)
            cashflow_df['cash_deflated_days'] = cashflow_df.apply(
                lambda x: (x['date'] - date).days * x['cash_deflated'] / 365, axis=1)
            duration = cashflow_df['cash_deflated_days'].sum() / cashflow_df['cash_deflated'].sum()
            if DURARION_TYPE == 'Macaulay':
                pass
            elif DURARION_TYPE == 'Modified':
                duration = duration / (1 + ytm / 100)
            else:
                raise NotImplementedError("Unknown DURARION_TYPE")
            return duration
        else:
            return 0

    def curve_to_duration(self, date, curve_df, DURARION_TYPE):
        ytm = self.curve_to_ytm(date, curve_df)
        duration = self.ytm_to_duration(date, ytm, DURARION_TYPE)
        return duration

    def get_profit(self, initial_date, end_date, begin_ytm, end_ytm):
        # print(self.code)
        buy_price = self.ytm_to_dirtyprice(initial_date, begin_ytm)
        sell_price = self.ytm_to_dirtyprice(end_date, end_ytm)
        cashflow = self._cashflow_df.copy()

        cashflow = cashflow[(cashflow['date'] > initial_date) & (cashflow['date'] <= end_date)]
        cashflow.loc['buy'] = [initial_date, -buy_price]
        cashflow.loc['sell'] = [end_date, sell_price]
        if end_date == self.end_date:
            cashflow.loc['adjust'] = [end_date, -100]
        profit = cashflow['cash'].sum()
        yield_simple = (profit / buy_price) / (end_date - initial_date).days * 365
        ts = [(d - initial_date).days / 365 for d in cashflow.date.to_list()]
        cf = cashflow.cash.to_list()

        # 年化收益大于100%的按100%，小于-100%的按-100%算
        if yield_simple <= -1:
            yield_simple = -1
            yield_compound = -1
        elif yield_simple >= 1:
            yield_simple = 1
            yield_compound = 1
        else:
            yield_compound = optimize.brentq(f_npv,
                                             a=-0.99,  # f(a) and f(b) must have opposite signs
                                             b=2,
                                             xtol=1e-8,
                                             args=(ts, cf),
                                             disp=True
                                             )

            # y1 = -1
            # y2 = 2
            # maxiter = 100
            # for _ in range(maxiter):
            #     yield_compound = (y1 + y2) / 2
            #     npv = f_npv(yield_compound, ts, cf)
            #     if abs(npv) < 1e-8:
            #         break
            #     if npv < 0:
            #         y2 = yield_compound
            #     else:
            #         y1 = yield_compound
        # print(f'yield_comp:{yield_compound}')
        # print(f'yield_simple:{yield_simple}')
        return profit, yield_compound * 100, yield_simple * 100


def produce_standard_bond(date, year, coupon_rate, issue_price=100, coupon_type='附息', coupon_frequency=1):
    code = '{}-{}'.format(datetime.datetime.strftime(date, '%Y%m%d'), year)
    initial_date = date
    end_date = date + relativedelta(years=year)
    standard_bond = Bond(code, initial_date, end_date, issue_price, coupon_rate, coupon_type, coupon_frequency)
    return standard_bond


if __name__ == '__main__':
    # code = '219915'
    # initial_date = datetime.datetime(2021, 4, 6)
    # end_date = datetime.datetime(2021, 10, 5)
    # issue_price = 98.9240
    # coupon_rate = 3.27
    # coupon_type = '贴现'
    # coupon_frequency = 0
    # a = Bond(code, initial_date, end_date, issue_price, coupon_rate, coupon_type, coupon_frequency)
    # date = datetime.datetime(2021, 9, 30)
    # cleanprice = 98.9620
    # print(a.cleanprice_to_ytm(date,cleanprice))
    date = datetime.datetime(2021, 12, 21)
    year = 1
    coupon_rate = 2.5
    bond = produce_standard_bond(date, year, coupon_rate, coupon_type='到期一次还本付息', coupon_frequency=1)
    print(bond.get_cashflow(date))

    # print(a.accrued_interest(date))
    #
    # print(a.ytm_to_dirtyprice(date, ytm))
    # print(a.ytm_to_cleanprice(date, ytm))
    #
    # print(a.dirtyprice_to_ytm(date, dirtyprice))
    # print(a.dirtyprice_to_cleanprice(date, dirtyprice))  ###
    #
    # print(a.cleanprice_to_ytm(date, cleanprice))
    # print(a.cleanprice_to_dirtyprice(date, cleanprice))
    #
    # # x=pd.DataFrame([[1],[2]],columns=['123'])
    # # x['111']=[1,2]
    # # print(x)
    # x = pd.DataFrame([], columns=['1', '2'])
    # x = x.append([1, 2])
    # print(x)
