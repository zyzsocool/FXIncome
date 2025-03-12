import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fxincome import const
from fxincome.utils import JsonModel
from fxincome import logging, logger, f_logger
from fxincome.spread.predict_spread import predict_pair_spread
import datetime


class BondData(bt.feeds.PandasData):
    """ """

    # Lines besides those required by backtrader (datetime, OHLC, volume, openinterest).
    lines = (
        "ytm",
        "matu",
        "out_bal",
        "code",
    )

    params = (
        ("datetime", "DATE"),
        ("close", "CLOSE"),
        ("open", "OPEN"),
        ("high", "HIGH"),
        ("low", "LOW"),
        ("volume", "VOL"),
        ("openinterest", -1),
        ("ytm", "YTM"),
        ("matu", "MATU"),
        ("out_bal", "OUT_BAL"),
        ("code", "CODE"),
    )


class IndexStrategy(bt.Strategy):
    SIZE = 6e6  # Bond Size. 100 face value per unit size. 6 million size -> 600 million face value.
    CASH_AVAILABLE = 630e6  # If use more than available cash, you need to borrow cash.
    INIT_CASH = 10e8  # Maximum cash for backtrader. Order will be rejected if cash is insufficient.

    params = (
        ("year", None),
        ("base_code", None),
        ("code_list", None),
        ("each_result_df", None),
        ("min_percentile", 25),
        ("max_percentile", 75),
        ("min_volume", 1e9),
    )

    def __init__(self):
        self.data = self.getdatabyname(self.p.base_code)
        self.result = self.p.each_result_df
        self.result["DATE"] = self.result.index
        self.result["DATE"] = self.result["DATE"].apply(lambda x: x.date())
        self.result["Yield"] = 0.0
        self.result["BaseYield"] = 0.0
        self.last_day = self.data.datetime.date(
            0
        )  # In init(), date(0) is the last day.
        self.current_size = [0, 0, 0]
        self.code_list_3 = []
        self.code_list_5 = []
        self.code_list_7 = []
        self.numbers_tradays = 0
        self.row_data = pd.read_excel(
            const.INDEX_ENHANCEMENT.CDB_YC_PATH, parse_dates=["DATE"]
        )
        # 只取前三年的数据
        self.row_data["DATE"] = self.row_data["DATE"].apply(lambda x: x.date())
        self.pass_data = self.row_data[
            (self.row_data["DATE"] >= datetime.date(self.p.year - 3, 1, 1))
            & (self.row_data["DATE"] <= datetime.date(self.p.year - 1, 12, 31))
        ]
        self.spread5_3_min = np.percentile(
            self.pass_data["5年-3年国开/均值"], self.p.min_percentile
        )
        self.spread7_5_min = np.percentile(
            self.pass_data["7年-5年国开/均值"], self.p.min_percentile
        )
        self.spread7_3_min = np.percentile(
            self.pass_data["7年-3年国开/均值"], self.p.min_percentile
        )
        self.spread5_3_max = np.percentile(
            self.pass_data["5年-3年国开/均值"], self.p.max_percentile
        )
        self.spread7_5_max = np.percentile(
            self.pass_data["7年-5年国开/均值"], self.p.max_percentile
        )
        self.spread7_3_max = np.percentile(
            self.pass_data["7年-3年国开/均值"], self.p.max_percentile
        )
        # 计算row_data中每天的"基准"的收益率,并将其存入self.result中
        self.now_data = self.row_data[
            (self.row_data["DATE"] >= datetime.date(self.p.year, 1, 1))
            & (self.row_data["DATE"] <= datetime.date(self.p.year, 12, 31))
        ]
        self.yield_base = 0
        for i in range(1, len(self.now_data)):
            self.result.loc[
                self.result["DATE"] == self.now_data["DATE"].iloc[i], "BaseYield"
            ] = (
                (self.now_data["基准"].iloc[i] - self.now_data["基准"].iloc[0])
                / self.now_data["基准"].iloc[0]
                * 100
            )
        # 计算最后一天的收益
        self.last_yield_base = (
            self.row_data.loc[self.row_data["DATE"] == self.last_day, "基准"].values[0]
            - self.row_data.loc[
                self.row_data["DATE"] == self.data.datetime.date(1), "基准"
            ].values[0]
        ) / self.row_data.loc[
            self.row_data["DATE"] == self.data.datetime.date(1), "基准"
        ].values[
            0
        ]

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.data.datetime.date(0)
        print(f"{dt:%Y%m%d} - {txt}")

    def _process_coupon(self, code):
        """
        This method is supposed to be called in next().
        It processes lending fee and coupon payment at the end of each day.
        (1)Lending fee is based on today's position and rate. Cash change at T+1. Only fee for borrowing is considered.
        (2)Coupon payment is based on T+1's actual coupon. Cash change at T+1.
        """
        cash_flow = pd.read_csv(
            const.INDEX_ENHANCEMENT.CDB_PATH + f"cash_flow_{code}.csv",
            parse_dates=["DATE"],
        )
        cash_flow["DATE"] = cash_flow["DATE"].dt.date
        today_remaining_payment_times = len(
            cash_flow[cash_flow["DATE"] > self.getdatabyname(code).datetime.date(0)]
        )
        tomorrow_remaining_payment_times = len(
            cash_flow[cash_flow["DATE"] > self.getdatabyname(code).datetime.date(1)]
        )
        if tomorrow_remaining_payment_times < today_remaining_payment_times:
            coupon_row_num = len(cash_flow) - today_remaining_payment_times
            # Bond holder will receive coupon payments at T+1.
            # Negative position means we short sell leg1, so we need to pay coupon to the lender.
            # Positive position means we hold leg1, so we receive coupon.
            coupon = (
                cash_flow.iloc[coupon_row_num]["AMOUNT"]
                * self.getposition(self.getdatabyname(code)).size
            )
            self.log(
                f"coupon payment {code} {coupon}",
                dt=self.getdatabyname(code).datetime.date(1),
            )
            self.broker.add_cash(coupon)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"Order {order.ref}, BUY EXECUTED, {order.data._name},year,{self.getdatabyname(order.data._name).matu[0]:.2f},{order.executed.price:.2f}, {order.executed.size:.2f}"
                )

            elif order.issell():
                self.log(
                    f"Order {order.ref}, SELL EXECUTED, {order.data._name},year,{self.getdatabyname(order.data._name).matu[0]:.2f},{order.executed.price:.2f}, {order.executed.size:.2f}"
                )

        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected,
            order.Expired,
        ]:
            self.log(f"Order {order.ref} Canceled/Margin/Rejected/Expired")

    def last_day_process(self):
        self.record()
        # 计算result中"cash_occured"不为0行的均值
        # self.mean_cash_occured = self.result[self.result['cash_occured']!=0]['cash_occured'].mean()
        # self.log(f'mean_cash_occred{self.mean_cash_occured}')
        self.log(
            f"Final Portfolio Value: {(self.broker.getvalue()-self.INIT_CASH)/self.CASH_AVAILABLE*100},base yield:{self.last_yield_base*100}"
        )
        self.log(f"numbers_tradays:{self.numbers_tradays}")
        return

    def record(self):
        self.result.loc[self.result["DATE"] == self.data.datetime.date(0), "Yield"] = (
            (self.broker.getvalue() - self.INIT_CASH) / self.CASH_AVAILABLE * 100
        )
        if (self.broker.getvalue() - self.INIT_CASH) / self.CASH_AVAILABLE * 100 < -1:
            self.log(
                f"Touching loss: {(self.broker.getvalue()-self.INIT_CASH)/self.CASH_AVAILABLE*100}"
            )
        if self.data.datetime.date(0) != self.last_day:
            self.broker.add_cash(
                (self.CASH_AVAILABLE - (self.INIT_CASH - self.broker.getcash()))
                * 0.02
                / 365
                * (self.data.datetime.date(1) - self.data.datetime.date(0)).days
            )


class IndexEnhancedStrategy(IndexStrategy):
    def __init__(self):
        super().__init__()

    def sell_bond(self, amount_to_sell, sell_list, code_list):
        code_all = code_list.copy()
        if amount_to_sell < self.getposition(self.getdatabyname(sell_list[0])).size:
            self.sell(data=self.getdatabyname(sell_list[0]), size=amount_to_sell)
        elif amount_to_sell == self.getposition(self.getdatabyname(sell_list[0])).size:
            self.sell(data=self.getdatabyname(sell_list[0]), size=amount_to_sell)
            code_list.remove(sell_list[0])
        else:
            for bond in code_all:
                bond_size = self.getposition(self.getdatabyname(bond)).size
                if bond_size <= amount_to_sell:
                    self.sell(data=self.getdatabyname(bond), size=bond_size)
                    amount_to_sell -= bond_size
                    code_list.remove(bond)
                else:
                    self.sell(data=self.getdatabyname(bond), size=amount_to_sell)
                    amount_to_sell = 0
                if amount_to_sell == 0:
                    break
        sell_list.remove(sell_list[0])
        return code_list, sell_list

    # strategy1
    def strategy_volume(
        self, delta_size, each_size, sell_list, years_3_bond, years_5_bond, years_7_bond
    ):
        if delta_size[0] > 0:
            # 买入year_3_bond中成交量最大的债券
            max_volume = 0
            max_volume_3_year_bond = ""
            for code in years_3_bond:
                if self.getdatabyname(code).volume[0] > max_volume:
                    max_volume = self.getdatabyname(code).volume[0]
                    max_volume_3_year_bond = code
            self.buy(
                data=self.getdatabyname(max_volume_3_year_bond), size=delta_size[0]
            )
            self.current_size[0] = each_size[0]
            # 如果max_volume_3_year_bond不在code_list_3中，则加入
            if max_volume_3_year_bond not in self.code_list_3:
                self.code_list_3.append(max_volume_3_year_bond)
        elif delta_size[0] < 0:
            self.code_list_3, sell_list = self.sell_bond(
                abs(delta_size[0]), sell_list, self.code_list_3
            )
            self.current_size[0] = each_size[0]
        if delta_size[1] > 0:
            # 买入5年期国开债
            max_volume = 0
            max_volume_5_year_bond = ""
            for code in years_5_bond:
                if self.getdatabyname(code).volume[0] > max_volume:
                    max_volume = self.getdatabyname(code).volume[0]
                    max_volume_5_year_bond = code
            self.buy(
                data=self.getdatabyname(max_volume_5_year_bond), size=delta_size[1]
            )
            self.current_size[1] = each_size[1]
            if max_volume_5_year_bond not in self.code_list_5:
                self.code_list_5.append(max_volume_5_year_bond)
        elif delta_size[1] < 0:
            self.code_list_5, sell_list = self.sell_bond(
                abs(delta_size[1]), sell_list, self.code_list_5
            )
            self.current_size[1] = each_size[1]
        if delta_size[2] > 0:
            # 买入7年期国开债
            max_volume = 0
            max_volume_7_year_bond = ""
            for code in years_7_bond:
                if self.getdatabyname(code).volume[0] > max_volume:
                    max_volume = self.getdatabyname(code).volume[0]
                    max_volume_7_year_bond = code
            self.buy(
                data=self.getdatabyname(max_volume_7_year_bond), size=delta_size[2]
            )
            self.current_size[2] = each_size[2]
            if max_volume_7_year_bond not in self.code_list_7:
                self.code_list_7.append(max_volume_7_year_bond)
        elif delta_size[2] < 0:
            self.code_list_7, sell_list = self.sell_bond(
                abs(delta_size[2]), sell_list, self.code_list_7
            )
            self.current_size[2] = each_size[2]

    # strategy2
    def strategy_closet(
        self,
        delta_size,
        each_size,
        sell_list,
        years_3_bond,
        years_5_bond,
        years_7_bond,
        yield_3,
        yield_5,
        yield_7,
    ):
        if delta_size[0] > 0:
            # 买入year_3_bond中与基准收益率最接近的债券
            closest_to_3_years_bond = min(
                years_3_bond,
                key=lambda bond: abs(self.getdatabyname(bond).ytm[0] - yield_3),
            )
            self.buy(
                data=self.getdatabyname(closest_to_3_years_bond), size=delta_size[0]
            )
            self.current_size[0] = each_size[0]
            if closest_to_3_years_bond not in self.code_list_3:
                self.code_list_3.append(closest_to_3_years_bond)
        elif delta_size[0] < 0:
            self.code_list_3, sell_list = self.sell_bond(
                abs(delta_size[0]), sell_list, self.code_list_3
            )
            self.current_size[0] = each_size[0]
        if delta_size[1] > 0:
            # 买入5年期国开债
            closest_to_5_years_bond = min(
                years_5_bond,
                key=lambda bond: abs(self.getdatabyname(bond).ytm[0] - yield_5),
            )
            self.buy(
                data=self.getdatabyname(closest_to_5_years_bond), size=delta_size[1]
            )
            self.current_size[1] = each_size[1]
            if closest_to_5_years_bond not in self.code_list_5:
                self.code_list_5.append(closest_to_5_years_bond)
        elif delta_size[1] < 0:
            self.code_list_5, sell_list = self.sell_bond(
                abs(delta_size[1]), sell_list, self.code_list_5
            )
            self.current_size[1] = each_size[1]
        if delta_size[2] > 0:
            # 买入7年期国开债
            closest_to_7_years_bond = min(
                years_7_bond,
                key=lambda bond: abs(self.getdatabyname(bond).ytm[0] - yield_7),
            )
            self.buy(
                data=self.getdatabyname(closest_to_7_years_bond), size=delta_size[2]
            )
            self.current_size[2] = each_size[2]
            if closest_to_7_years_bond not in self.code_list_7:
                self.code_list_7.append(closest_to_7_years_bond)
        elif delta_size[2] < 0:
            self.code_list_7, sell_list = self.sell_bond(
                abs(delta_size[2]), sell_list, self.code_list_7
            )
            self.current_size[2] = each_size[2]

    # strategy3
    def strategy_closet_year(
        self, delta_size, each_size, sell_list, years_3_bond, years_5_bond, years_7_bond
    ):
        if delta_size[0] > 0:
            # 买入year_3_bond中与基准收益率最接近的债券
            closest_to_3_years_bond = min(
                years_3_bond, key=lambda bond: abs(self.getdatabyname(bond).matu[0] - 3)
            )
            self.buy(
                data=self.getdatabyname(closest_to_3_years_bond), size=delta_size[0]
            )
            self.current_size[0] = each_size[0]
            if closest_to_3_years_bond not in self.code_list_3:
                self.code_list_3.append(closest_to_3_years_bond)
        elif delta_size[0] < 0:
            self.code_list_3, sell_list = self.sell_bond(
                abs(delta_size[0]), sell_list, self.code_list_3
            )
            self.current_size[0] = each_size[0]
        if delta_size[1] > 0:
            # 买入5年期国开债
            closest_to_5_years_bond = min(
                years_5_bond, key=lambda bond: abs(self.getdatabyname(bond).matu[0] - 5)
            )
            self.buy(
                data=self.getdatabyname(closest_to_5_years_bond), size=delta_size[1]
            )
            self.current_size[1] = each_size[1]
            if closest_to_5_years_bond not in self.code_list_5:
                self.code_list_5.append(closest_to_5_years_bond)
        elif delta_size[1] < 0:
            self.code_list_5, sell_list = self.sell_bond(
                abs(delta_size[1]), sell_list, self.code_list_5
            )
            self.current_size[1] = each_size[1]
        if delta_size[2] > 0:
            # 买入7年期国开债
            closest_to_7_years_bond = min(
                years_7_bond, key=lambda bond: abs(self.getdatabyname(bond).matu[0] - 7)
            )
            self.buy(
                data=self.getdatabyname(closest_to_7_years_bond), size=delta_size[2]
            )
            self.current_size[2] = each_size[2]
            if closest_to_7_years_bond not in self.code_list_7:
                self.code_list_7.append(closest_to_7_years_bond)
        elif delta_size[2] < 0:
            self.code_list_7, sell_list = self.sell_bond(
                abs(delta_size[2]), sell_list, self.code_list_7
            )
            self.current_size[2] = each_size[2]

    # strategy4
    def strategy_max_yield(
        self, delta_size, each_size, sell_list, years_3_bond, years_5_bond, years_7_bond
    ):
        if delta_size[0] > 0:
            # 买入year_3_bond中收益率最高的债券
            max_yield_3_bond = max(
                years_3_bond, key=lambda bond: self.getdatabyname(bond).ytm[0]
            )
            self.buy(data=self.getdatabyname(max_yield_3_bond), size=delta_size[0])
            self.current_size[0] = each_size[0]
            if max_yield_3_bond not in self.code_list_3:
                self.code_list_3.append(max_yield_3_bond)
        elif delta_size[0] < 0:
            self.code_list_3, sell_list = self.sell_bond(
                abs(delta_size[0]), sell_list, self.code_list_3
            )
            self.current_size[0] = each_size[0]
        if delta_size[1] > 0:
            # 买入5年期国开债
            max_yield_5_bond = max(
                years_5_bond, key=lambda bond: self.getdatabyname(bond).ytm[0]
            )
            self.buy(data=self.getdatabyname(max_yield_5_bond), size=delta_size[1])
            self.current_size[1] = each_size[1]
            if max_yield_5_bond not in self.code_list_5:
                self.code_list_5.append(max_yield_5_bond)
        elif delta_size[1] < 0:
            self.code_list_5, sell_list = self.sell_bond(
                abs(delta_size[1]), sell_list, self.code_list_5
            )
            self.current_size[1] = each_size[1]
        if delta_size[2] > 0:
            # 买入7年期国开债
            max_yield_7_bond = max(
                years_7_bond, key=lambda bond: self.getdatabyname(bond).ytm[0]
            )
            self.buy(data=self.getdatabyname(max_yield_7_bond), size=delta_size[2])
            self.current_size[2] = each_size[2]
            if max_yield_7_bond not in self.code_list_7:
                self.code_list_7.append(max_yield_7_bond)
        elif delta_size[2] < 0:
            self.code_list_7, sell_list = self.sell_bond(
                abs(delta_size[2]), sell_list, self.code_list_7
            )
            self.current_size[2] = each_size[2]

    def prenext(self):
        # 在result中记录每天每个资产的持仓
        sum_position = 0
        for code in self.p.code_list:
            if len(self.getdatabyname(code)) != 0:
                self.result.loc[
                    self.result["DATE"] == self.data.datetime.date(0), code
                ] = self.getposition(self.getdatabyname(code)).size
                sum_position += self.getposition(self.getdatabyname(code)).size
        self.result.loc[
            self.result["DATE"] == self.data.datetime.date(0), "sum_position"
        ] = sum_position
        if self.data.datetime.date(0) == self.last_day:
            return self.last_day_process()
        # 计算所持有债券今天的票息
        for code in self.p.code_list:
            if len(self.getdatabyname(code)) == 0:
                continue
            if self.getdatabyname(code).datetime.date(0) == self.last_day:
                continue
            if self.getposition(self.getdatabyname(code)).size > 0:
                self._process_coupon(code)
        # 判断阈值
        each_size = [0, 0, 0]
        spread7_5 = self.row_data.loc[
            self.row_data["DATE"] == self.data.datetime.date(0), "7年-5年国开/均值"
        ].values[0]
        spread7_3 = self.row_data.loc[
            self.row_data["DATE"] == self.data.datetime.date(0), "7年-3年国开/均值"
        ].values[0]
        spread5_3 = self.row_data.loc[
            self.row_data["DATE"] == self.data.datetime.date(0), "5年-3年国开/均值"
        ].values[0]
        if spread7_5 < self.spread7_5_min:
            each_size[1] = each_size[1] + 2
        elif spread7_5 > self.spread7_5_max:
            each_size[2] = each_size[2] + 2
        else:
            each_size[1] = each_size[1] + 1
            each_size[2] = each_size[2] + 1
        if spread5_3 < self.spread5_3_min:
            each_size[0] = each_size[0] + 2
        elif spread5_3 > self.spread5_3_max:
            each_size[1] = each_size[1] + 2
        else:
            each_size[0] = each_size[0] + 1
            each_size[1] = each_size[1] + 1
        if spread7_3 < self.spread7_3_min:
            each_size[0] = each_size[0] + 2
        elif spread7_3 > self.spread7_3_max:
            each_size[2] = each_size[2] + 2
        else:
            each_size[0] = each_size[0] + 1
            each_size[2] = each_size[2] + 1
        # 将each_size中的值扩大self.SIZE/6倍
        each_size = [i * self.SIZE / 6 for i in each_size]
        delta_size = [each_size[i] - self.current_size[i] for i in range(3)]
        # self.log(f"current_size:{self.current_size}")
        # self.log(f"each_size:{each_size}")
        # self.log(f"delta_size:{delta_size}")
        # 如果delta_size中的值都为0，则不进行操作
        if delta_size == [0, 0, 0]:
            # self.log("No operation")
            self.record()
            return
        else:
            self.numbers_tradays += 1
        # 从row_data中找到今日3年国开和5年国开的收益率
        yield_3 = self.row_data.loc[
            self.row_data["DATE"] == self.data.datetime.date(0), "3年国开"
        ].values[0]
        yield_5 = self.row_data.loc[
            self.row_data["DATE"] == self.data.datetime.date(0), "5年国开"
        ].values[0]
        yield_7 = self.row_data.loc[
            self.row_data["DATE"] == self.data.datetime.date(0), "7年国开"
        ].values[0]
        # 筛选出所有收益率上下浮动在0.05以内的债券
        years_3_bond = [
            code
            for code in self.p.code_list
            if len(self.getdatabyname(code)) != 0
            and abs(self.getdatabyname(code).ytm[0] - yield_3) < 0.05
        ]
        years_5_bond = [
            code
            for code in self.p.code_list
            if len(self.getdatabyname(code)) != 0
            and abs(self.getdatabyname(code).ytm[0] - yield_5) < 0.05
        ]
        years_7_bond = [
            code
            for code in self.p.code_list
            if len(self.getdatabyname(code)) != 0
            and abs(self.getdatabyname(code).ytm[0] - yield_7) < 0.05
        ]
        # 筛选出期限在2-4年，4-6年，6-8年的债券
        years_3_bond = [
            code
            for code in years_3_bond
            if self.getdatabyname(code).matu[0] > 2
            and self.getdatabyname(code).matu[0] < 3.5
        ]
        years_5_bond = [
            code
            for code in years_5_bond
            if self.getdatabyname(code).matu[0] > 4
            and self.getdatabyname(code).matu[0] < 5.5
        ]
        years_7_bond = [
            code
            for code in years_7_bond
            if self.getdatabyname(code).matu[0] > 6
            and self.getdatabyname(code).matu[0] < 7.5
        ]
        # 筛选出所有volume大于min_volume的datas
        years_3_bond = [
            code
            for code in years_3_bond
            if self.getdatabyname(code).volume[0] > self.p.min_volume
        ]
        years_5_bond = [
            code
            for code in years_5_bond
            if self.getdatabyname(code).volume[0] > self.p.min_volume
        ]
        years_7_bond = [
            code
            for code in years_7_bond
            if self.getdatabyname(code).volume[0] > self.p.min_volume
        ]
        # 判断是否能买卖债券，0表示可以买卖，大于0表示不可以。
        # 当delta_size[i] > 0时，判断是否有可买债券，如果没有，则feasibility_judge_buy[i] = 3/5/7
        # 当delta_size[i] < 0时，判断所持仓债券中是否有可卖债券，如果有，则feasibility_judge_sell[i] = 0
        feasibility_judge_buy = [0, 0, 0]
        feasibility_judge_sell = [3, 5, 7]
        sell_list = []
        if delta_size[0] > 0:
            feasibility_judge_sell[0] = 0
            if len(years_3_bond) == 0:
                feasibility_judge_buy[0] = 3
        elif delta_size[0] < 0:
            for code in self.code_list_3:
                if self.getdatabyname(code).volume[0] >= self.p.min_volume:
                    sell_list.append(code)
                    feasibility_judge_sell[0] = 0
                    break
        else:
            feasibility_judge_sell[0] = 0
        if delta_size[1] > 0:
            feasibility_judge_sell[1] = 0
            if len(years_5_bond) == 0:
                feasibility_judge_buy[1] = 5
        elif delta_size[1] < 0:
            for code in self.code_list_5:
                if self.getdatabyname(code).volume[0] >= self.p.min_volume:
                    sell_list.append(code)
                    feasibility_judge_sell[1] = 0
                    break
        else:
            feasibility_judge_sell[1] = 0
        if delta_size[2] > 0:
            feasibility_judge_sell[2] = 0
            if len(years_7_bond) == 0:
                feasibility_judge_buy[2] = 7
        elif delta_size[2] < 0:
            for code in self.code_list_7:
                if self.getdatabyname(code).volume[0] >= self.p.min_volume:
                    sell_list.append(code)
                    feasibility_judge_sell[2] = 0
                    break
        else:
            feasibility_judge_sell[2] = 0
        if feasibility_judge_sell != [0, 0, 0] or feasibility_judge_buy != [0, 0, 0]:
            self.log(
                f"feasibility_judge_buy:{feasibility_judge_buy}, feasibility_judge_sell:{feasibility_judge_sell}is no pass"
            )
            self.record()
            self.numbers_tradays -= 1
            return
        self.strategy_max_yield(
            delta_size, each_size, sell_list, years_3_bond, years_5_bond, years_7_bond
        )
        self.record()


def plot_result(year):
    result = pd.read_csv(
        const.INDEX_ENHANCEMENT.RESULT_PATH + f"{year}_result.csv", parse_dates=["DATE"]
    )
    fig = plt.figure(num=1, figsize=(15, 5))
    ax = fig.add_subplot(111, label="1")
    # 画出yield和base yield
    lns1 = ax.plot(result["DATE"], result["Yield"], color="r", label="Yield")
    lns2 = ax.plot(result["DATE"], result["BaseYield"], color="b", label="BaseYield")
    lns = lns1 + lns2
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield")
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=10)
    plt.title(f"{year} Yield")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # use a dataframe to record the return of each year
    year_start = 2017
    year_end = 2023
    result_df = pd.DataFrame(
        index=range(year_start, year_end + 1),
        columns=[
            "Yield",
            "Base Yield",
            "Excess Return",
            "Max Loss",
            "Base Max Loss",
            "Trade days",
        ],
    )
    min_percentile = 25
    max_percentile = 75
    for year in range(year_start, year_end + 1):
        all_bond = pd.read_excel(
            const.INDEX_ENHANCEMENT.CDB_INFO_PATH,
            parse_dates=["maturitydate", "carrydate", "issuedate"],
        )
        selected_bond = all_bond[
            (all_bond["maturitydate"] >= datetime.datetime(year, 12, 31))
        ]
        selected_bond = selected_bond[
            (selected_bond["maturitydate"] - selected_bond["carrydate"]).dt.days > 380
        ]
        # 对selected_bond中的每只券，分别从D:\data中读取数据，进行回测
        cerebro = bt.Cerebro()
        code_list = []
        base_code = ""
        for i, j in selected_bond.iterrows():
            code = j["windcode"][:-3]
            # 如果code最后两位为QF，则跳过
            if code[-2:] == "QF":
                continue
            # 如果code开头前两位为year的后两位且base_code为空，则将code赋值给base_code
            if code[:2] == str(year - 1)[-2:] and base_code == "":
                base_code = j["windcode"][:-3]
            code_list.append(code)
            input_file = const.INDEX_ENHANCEMENT.CDB_PATH + code + ".csv"
            price_df = pd.read_csv(input_file, parse_dates=["DATE"])
            price_df = price_df[price_df["DATE"] <= datetime.datetime(year, 12, 31)]
            price_df = price_df[price_df["DATE"] >= datetime.datetime(year, 1, 1)]
            data1 = BondData(dataname=price_df, nocase=True)
            cerebro.adddata(data1, name=code)
        # 读取base_code的数据
        input_file = const.INDEX_ENHANCEMENT.CDB_PATH + base_code + ".csv"
        price_df = pd.read_csv(input_file, parse_dates=["DATE"])
        price_df = price_df[price_df["DATE"] <= datetime.datetime(year, 12, 31)]
        price_df = price_df[price_df["DATE"] >= datetime.datetime(year, 1, 1)]
        each_result_df = pd.DataFrame(
            index=price_df["DATE"],
            columns=code_list,
        )
        cerebro.addstrategy(
            IndexEnhancedStrategy,
            year=year,
            base_code=base_code,
            code_list=code_list,
            each_result_df=each_result_df,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
        )
        cerebro.broker.set_cash(IndexEnhancedStrategy.INIT_CASH)
        # cerebro.broker.set_slippage_perc(perc=0.0001)
        logger.info(year)
        strategies = cerebro.run()
        logger.info(
            f"PROFIT: {(cerebro.broker.get_value() - IndexEnhancedStrategy.INIT_CASH) / 10000:.2f}"
        )
        result_df.loc[year, "Yield"] = (
            (cerebro.broker.get_value() - IndexEnhancedStrategy.INIT_CASH)
            / strategies[0].CASH_AVAILABLE
            * 100
        )
        # 删除求和后为0的列
        strategies[0].result.to_csv(
            const.INDEX_ENHANCEMENT.RESULT_PATH + f"{year}_result.csv", index=False
        )
        result_df.loc[year, "Base Yield"] = strategies[0].last_yield_base * 100
        result_df.loc[year, "Excess Return"] = (
            result_df.loc[year, "Yield"] - result_df.loc[year, "Base Yield"]
        )
        result_df.loc[year, "Max Loss"] = strategies[0].result["Yield"].min()
        result_df.loc[year, "Base Max Loss"] = strategies[0].result["BaseYield"].min()
        result_df.loc[year, "Trade days"] = strategies[0].numbers_tradays
        # plot_result(year)
    # calulate the average return of all years
    result_df.loc["Average"] = result_df.mean()
    result_df.to_csv(const.INDEX_ENHANCEMENT.RESULT_PATH + "all_result.csv", index=True)
