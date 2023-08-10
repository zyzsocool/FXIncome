import datetime

import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt

from fxincome.const import PATH, SPREAD
from fxincome.utils import JsonModel
from fxincome import logging, logger, f_logger
from fxincome.spread.predict_spread import predict_pair_spread


class SpreadData(bt.feeds.PandasData):
    """
    Spread = leg2 ytm - leg1 ytm.  YTM's unit is %.
    The prices(OHLC) = Leg1's full price - Leg2's full price.
    Buy at price means buy leg1, sell leg2 with the same size. The net cash flow of buy is -price * size.
    Sell at price means sell leg1, buy leg2 with the same size. The net cash flow of sell is +price * size.
    The position of Spread is either >0, =0, or <0.
    Position > 0 means LONG spread, i.e. LONG leg1, SHORT leg2.
    Position < 0 means SHORT spread, i.e. SHORT leg1, LONG leg2.
    """

    # Lines besides those required by backtrader (datetime, OHLC, volume, openinterest).
    lines = (
        "code_leg1",
        "out_leg1",
        "ytm_leg1",
        "lend_rate_leg1",
        "code_leg2",
        "vol_leg2",
        "out_leg2",
        "ytm_leg2",
        "lend_rate_leg2",
        "spread",
        "spread_min",
    )

    params = (
        ("datetime", "DATE"),
        ("close", "CLOSE"),
        ("open", "OPEN"),
        ("high", "HIGH"),
        ("low", "LOW"),
        ("volume", "VOL_LEG1"),
        ("openinterest", -1),
        ("code_leg1", "CODE_LEG1"),
        ("out_leg1", "OUT_BAL_LEG1"),
        ("ytm_leg1", "YTM_LEG1"),
        ("lend_rate_leg1", "LEND_RATE_LEG1"),
        ("code_leg2", "CODE_LEG2"),
        ("vol_leg2", "VOL_LEG2"),
        ("out_leg2", "OUT_BAL_LEG2"),
        ("ytm_leg2", "YTM_LEG2"),
        ("lend_rate_leg2", "LEND_RATE_LEG2"),
        ("spread", "SPREAD"),
        ("spread_min", "SPREAD_MIN"),
    )


class SpreadStrategy(bt.Strategy):
    SIZE = 1e6  # 1 million size for 100 million face value of bond
    INIT_CASH = 1e7  # 10 million initial cash for 100 million face value of bond

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
        self.leg1_code = str(int(self.data.code_leg1[0]))
        self.cash_flow_leg1 = pd.read_csv(
            PATH.SPREAD_DATA + f"cash_flow_{self.leg1_code}.csv", parse_dates=["DATE"]
        )
        self.cash_flow_leg1["DATE"] = self.cash_flow_leg1["DATE"].dt.date
        self.leg2_code = str(int(self.data.code_leg2[0]))
        self.cash_flow_leg2 = pd.read_csv(
            PATH.SPREAD_DATA + f"cash_flow_{self.leg2_code}.csv", parse_dates=["DATE"]
        )
        self.cash_flow_leg2["DATE"] = self.cash_flow_leg2["DATE"].dt.date
        self.result = pd.read_csv(
            PATH.SPREAD_DATA + f"{self.leg1_code}_{self.leg2_code}_bt.csv",
            parse_dates=["DATE"],
        )
        self.result["DATE"] = self.result["DATE"].dt.date
        self.result["Profit"] = 0.0
        self.result["TotalFee"] = 0.0
        self.result["Sell"] = 0.0
        self.result["Buy"] = 0.0
        self.last_day = self.data.datetime.date(
            0
        )  # In init(), date(0) is the last day.

    def log(self, txt, dt=None, legs=None, level=logging.DEBUG):
        dt = dt or self.getdatabyname(self.ST_NAME).datetime.date(0)
        legs = legs or f"{self.leg1_code}_{self.leg2_code}"
        f_logger.log(level, f"{legs} - {dt:%Y%m%d} - {txt}")

    def _calculate_lend_fee(
        self, face_value: int, rate: float, direction: str
    ) -> float:
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
        if direction == "borrow":
            fee = (
                -rate
                * abs(face_value)
                / 365
                * (self.data.datetime.date(1) - self.data.datetime.date(0)).days
            )
        elif direction == "lend":
            fee = (
                rate
                * abs(face_value)
                / 365
                * (self.data.datetime.date(1) - self.data.datetime.date(0)).days
            )
        else:
            raise ValueError("direction must be 'borrow' or 'lend'")
        return fee

    def _process_fee_coupon(self):
        """
        This method is supposed to be called in next().
        It processes lending fee and coupon payment at the end of each day.
        (1)Lending fee is based on today's position and rate. Cash change at T+1. Only fee for borrowing is considered.
        (2)Coupon payment is based on T+1's actual coupon. Cash change at T+1.
        """
        lend_fee = 0.0
        if self.getposition(self.data).size < 0:
            lend_fee = self._calculate_lend_fee(
                face_value=self.getposition(self.data).size * 100,
                rate=self.lend_rate_leg1[0],
                direction="borrow",
            )
        elif self.getposition(self.data).size > 0:
            lend_fee = self._calculate_lend_fee(
                face_value=self.getposition(self.data).size * 100,
                rate=self.lend_rate_leg2[0],
                direction="borrow",
            )
        self.broker.add_cash(
            lend_fee
        )  # Lending fee for borrowing is negative. broker.add_cash() takes effect at T+1.
        self.total_fee += lend_fee  # Total fee is accumulated up to now.
        self.result.loc[
            self.result["DATE"] == self.data.datetime.date(0), "TotalFee"
        ] = self.total_fee

        # Coupon payment dates of cash_flow are dates when we receive coupon payments.
        # We receive coupon payments at T+1 only when we hold the bond at T.
        # Coupon payment dates are sometimes not trading days. Backtrader datafeed's dates are all trading days.
        # We compute the left payment times to determine whether we receive coupon payments at T+1.
        today_leg1_remaining_payment_times = len(
            self.cash_flow_leg1[
                self.cash_flow_leg1["DATE"] > self.data.datetime.date(0)
            ]
        )
        today_leg2_remaining_payment_times = len(
            self.cash_flow_leg2[
                self.cash_flow_leg2["DATE"] > self.data.datetime.date(0)
            ]
        )
        tomorrow_leg1_remaining_payment_times = len(
            self.cash_flow_leg1[
                self.cash_flow_leg1["DATE"] > self.data.datetime.date(1)
            ]
        )
        tomorrow_leg2_remaining_payment_times = len(
            self.cash_flow_leg2[
                self.cash_flow_leg2["DATE"] > self.data.datetime.date(1)
            ]
        )
        if tomorrow_leg1_remaining_payment_times < today_leg1_remaining_payment_times:
            coupon_row_num = (
                len(self.cash_flow_leg1) - today_leg1_remaining_payment_times
            )
            # Leg1's holder will receive coupon payments at T+1.
            # Negative position means we short sell leg1, so we need to pay coupon to the lender.
            # Positive position means we hold leg1, so we receive coupon.
            coupon = (
                self.cash_flow_leg1.iloc[coupon_row_num]["AMOUNT"]
                * self.getposition(self.data).size
            )
            self.log(f"coupon payment {coupon}", dt=self.data.datetime.date(1))
            self.broker.add_cash(coupon)
        if tomorrow_leg2_remaining_payment_times < today_leg2_remaining_payment_times:
            coupon_row_num = (
                len(self.cash_flow_leg2) - today_leg2_remaining_payment_times
            )
            # Leg2's holder will receive coupon payments at T+1.
            # Negative position means we hold leg2, so we receive coupon.
            # Positive position means we short sell leg2, so we need to pay coupon to the lender.
            coupon = (
                -self.cash_flow_leg2.iloc[coupon_row_num]["AMOUNT"]
                * self.getposition(self.data).size
            )
            self.log(f"coupon payment {coupon}", dt=self.data.datetime.date(1))
            self.broker.add_cash(coupon)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"Order {order.ref}, BUY EXECUTED, {order.executed.price:.2f}, {order.executed.size:.2f}"
                )
                self.result.loc[
                    self.result["DATE"] == self.data.datetime.date(0), "Buy"
                ] = 1
            elif order.issell():
                self.log(
                    f"Order {order.ref}, SELL EXECUTED, {order.executed.price:.2f}, {order.executed.size:.2f}"
                )
                self.result.loc[
                    self.result["DATE"] == self.data.datetime.date(0), "Sell"
                ] = 1
        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected,
            order.Expired,
        ]:
            self.log(f"Order {order.ref} Canceled/Margin/Rejected/Expired")


class BaselineStrategy(SpreadStrategy):
    ST_NAME = "baseline"

    def __init__(self):
        super().__init__()

    def next(self):
        # The last day has no tomorrow.
        if self.data.datetime.date(0) == self.last_day:
            self.result.loc[
                self.result["DATE"] == self.data.datetime.date(0), "TotalFee"
            ] = self.total_fee
            self.result.loc[
                self.result["DATE"] == self.data.datetime.date(0), "Profit"
            ] = (self.broker.getvalue() - self.INIT_CASH)
            return

        self._process_fee_coupon()

        # trading logic
        condition1 = self.spread[0] >= -0.03
        condition2 = self.vol_leg2[0] < self.volume[0]
        condition3 = self.spread[0] > self.spread_min[0] * 0.9
        condition4 = self.getposition(self.data).size < 0
        condition5 = self.spread[0] >= -0.005
        condition6 = self.out_leg2[0] > self.out_leg1[0] * 0.8
        condition7 = self.out_leg2[0] < self.out_leg1[0] * 0.6
        condition8 = self.spread[0] <= -0.04
        condition9 = self.vol_leg2[0] - self.volume[0] > 1e11
        condition10 = self.getposition(self.data).size == 0
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
        self.result.loc[self.result["DATE"] == self.data.datetime.date(0), "Profit"] = (
            self.broker.getvalue() - self.INIT_CASH
        )


class PredictStrategy(SpreadStrategy):
    ST_NAME = "predict"

    def __init__(self):
        super().__init__()
        self.model_name_up = "spread_0.665_XGB_20230428_1020"
        self.model_attr = JsonModel.load_attr(
            self.model_name_up, PATH.SPREAD_MODEL + JsonModel.JSON_NAME
        )
        self.days_forward_up = self.model_attr.labels["LABEL"]["days_forward"]
        self.threshold_up = abs(self.model_attr.labels["LABEL"]["spread_threshold"])
        self.up_preds = predict_pair_spread(
            self.model_name_up, self.leg1_code, self.leg2_code
        )
        self.model_name_down = "spread_0.635_XGB_20230428_1029"
        self.model_attr = JsonModel.load_attr(
            self.model_name_down, PATH.SPREAD_MODEL + JsonModel.JSON_NAME
        )
        self.days_forward_down = self.model_attr.labels["LABEL"]["days_forward"]
        self.threshold_down = abs(self.model_attr.labels["LABEL"]["spread_threshold"])
        self.down_preds = predict_pair_spread(
            self.model_name_down, self.leg1_code, self.leg2_code
        )
        self.unit_size_up = self.SIZE / self.days_forward_up
        self.unit_size_down = self.SIZE / self.days_forward_down
        self.buy_records = list()  # a list of tuple (spread, date)
        self.long_position = 0  # Unit is the same as SIZE.
        self.sell_records = list()  # a list of tuple (spread, date)
        self.short_position = 0  # Unit is the same as SIZE.

    def next(self):
        today = self.data.datetime.date(0)
        up_preds = self.up_preds.query("DATE ==@today")
        down_preds = self.down_preds.query("DATE ==@today")
        # The last day has no tomorrow.
        if (
            self.data.datetime.date(0) == self.last_day
            or len(up_preds) == 0
            or len(down_preds) == 0
        ):
            self.result.loc[
                self.result["DATE"] == self.data.datetime.date(0), "TotalFee"
            ] = self.total_fee
            self.result.loc[
                self.result["DATE"] == self.data.datetime.date(0), "Profit"
            ] = (self.broker.getvalue() - self.INIT_CASH)
            return

        self._process_fee_coupon()

        # trading logic
        up_pred = int(up_preds.pred.iat[0])
        down_pred = int(down_preds.pred.iat[0])
        # check if we need to close position
        for buy_record in self.buy_records:
            prediction_correct = self.spread[0] >= buy_record[0] + self.threshold_up
            end_of_period = self.data.datetime.date(0) - buy_record[
                1
            ] == datetime.timedelta(days=self.days_forward_up)
            stop_loss = self.spread[0] <= buy_record[0] - 2 * self.threshold_up
            if (
                prediction_correct
                or (end_of_period and not up_pred == 1)
                or stop_loss
            ):
                self.sell(data=self.data, size=self.unit_size_up)
                self.buy_records.remove(buy_record)
                self.long_position -= self.unit_size_up
            elif end_of_period and up_pred == 1:
                # Update the cost of trade but hold position
                self.buy_records.remove(buy_record)
                self.buy_records.append((self.spread[0], self.data.datetime.date(0)))
        for sell_record in self.sell_records:
            prediction_correct = self.spread[0] <= sell_record[0] - self.threshold_down
            end_of_period = self.data.datetime.date(0) - sell_record[
                1
            ] == datetime.timedelta(days=self.days_forward_down)
            stop_loss = self.spread[0] >= sell_record[0] + 2 * self.threshold_down
            if (
                prediction_correct
                or (end_of_period and not down_pred == 1)
                or stop_loss
            ):
                self.buy(data=self.data, size=self.unit_size_down)
                self.sell_records.remove(sell_record)
                self.short_position -= self.unit_size_down
            elif end_of_period and down_pred == 1:
                # Update the cost of trade but hold position
                self.sell_records.remove(sell_record)
                self.sell_records.append((self.spread[0], self.data.datetime.date(0)))

        # # check if we need to open position
        if (up_pred == 1) and (self.long_position < self.SIZE):
            self.buy(data=self.data, size=self.unit_size_up)
            self.buy_records.append((self.spread[0], self.data.datetime.date(0)))
            self.long_position += self.unit_size_up
        if (down_pred == 1) and (self.short_position < self.SIZE):
            self.sell(data=self.data, size=self.unit_size_down)
            self.sell_records.append((self.spread[0], self.data.datetime.date(0)))
            self.short_position += self.unit_size_down

        self.result.loc[self.result["DATE"] == self.data.datetime.date(0), "Profit"] = (
            self.broker.getvalue() - self.INIT_CASH
        )


def plot_spread(leg1, leg2):
    input_file = PATH.SPREAD_DATA + f"{leg1}_{leg2}_result.csv"
    data = pd.read_csv(input_file, parse_dates=["DATE"])
    fig = plt.figure(num=1, figsize=(15, 5))
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    lns1 = ax.plot(data["DATE"], data["SPREAD"], color="r", label="SPREAD")
    lns = lns1
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread")
    for i in range(0, len(data["Buy"])):
        if data["Buy"][i] == 1:
            ax.annotate(
                "Buy",
                xy=(data["DATE"][i], data["SPREAD"][i]),
                xytext=(data["DATE"][i], data["SPREAD"][i] - 0.01),
                arrowprops=dict(facecolor="green", shrink=0.05),
            )
    for i in range(0, len(data["Sell"])):
        if data["Sell"][i] == 1:
            ax.annotate(
                "Sell",
                xy=(data["DATE"][i], data["SPREAD"][i]),
                xytext=(data["DATE"][i], data["SPREAD"][i] + 0.01),
                arrowprops=dict(facecolor="yellow", shrink=0.05),
            )
    lns2 = ax2.plot(data["DATE"], data["Profit"], label="Profit")
    lns += lns2
    ax2.set_ylabel("Profit")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.xaxis.set_label_position("top")
    ax2.yaxis.set_label_position("right")
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=10)
    plt.title(f"{leg1} and {leg2} Spread Trading")
    plt.tight_layout()
    plt.show()


def main():
    # Baseline
    # for i in range(0, 15):
    #     leg1_code = SPREAD.CDB_CODES[i]
    #     leg2_code = SPREAD.CDB_CODES[i + 1]
    #     cerebro = bt.Cerebro()
    #     input_file = PATH.SPREAD_DATA + leg1_code + '_' + leg2_code + '_bt.csv'
    #     price_df = pd.read_csv(input_file, parse_dates=['DATE'])
    #     # minimum spread in the past 15 days are used as the threshold to open a position
    #     price_df['SPREAD_MIN'] = price_df['SPREAD'].rolling(15).min()
    #     price_df.loc[:, ['SPREAD_MIN']] = price_df.loc[:, ['SPREAD_MIN']].fillna(method='backfill')
    #     price_df = price_df.dropna()
    #     data1 = SpreadData(dataname=price_df, nocase=True)
    #     cerebro.adddata(data1, name=BaselineStrategy.ST_NAME)
    #     cerebro.addstrategy(BaselineStrategy)
    #     cerebro.broker.set_cash(BaselineStrategy.INIT_CASH)
    #     logger.info(leg1_code + '_' + leg2_code)
    #     strategies=cerebro.run()
    #     logger.info(f'PROFIT: {(cerebro.broker.get_value() - BaselineStrategy.INIT_CASH ) / 10000:.2f}')
    #     strategies[0].result.to_csv(PATH.SPREAD_DATA + f'{leg1_code}_{leg2_code}_result.csv', index=False)
    # for i in range(0, 15):
    #     leg1_code = SPREAD.CDB_CODES[i]
    #     leg2_code = SPREAD.CDB_CODES[i + 1]
    #     plot_spread(leg1_code,leg2_code)

    # Prediction
    for i in range(11, 14):
        leg1_code = SPREAD.CDB_CODES[i]
        leg2_code = SPREAD.CDB_CODES[i + 1]
        cerebro = bt.Cerebro()
        input_file = PATH.SPREAD_DATA + leg1_code + "_" + leg2_code + "_bt.csv"
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
        logger.info(
            f"PROFIT: {(cerebro.broker.get_value() - PredictStrategy.INIT_CASH) / 10000:.2f}"
        )
        strategies[0].result.to_csv(
            PATH.SPREAD_DATA + f"{leg1_code}_{leg2_code}_result.csv", index=False
        )
    for i in range(11, 14):
        leg1_code = SPREAD.CDB_CODES[i]
        leg2_code = SPREAD.CDB_CODES[i + 1]
        plot_spread(leg1_code, leg2_code)


if __name__ == "__main__":
    main()
