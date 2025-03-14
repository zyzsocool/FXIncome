import pandas as pd
import numpy as np
import datetime
from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from fxincome import const


class IndexStrategy(StrategyTemplate):
        
    SIZE = 6e6  # Bond Size. 100 face value per unit size. 6 million size -> 600 million face value.
    CASH_AVAILABLE = 630e6  # If use more than available cash, you need to borrow cash.
    MAX_CASH = 10e8  # Maximum cash for backtrader. Order will be rejected if cash is insufficient.

    # Fixed parameters to be shown and set in the UI
    parameters = ["SIZE", "CASH_AVAILABLE", "MAX_CASH"]

    # Variables to be shown in the UI
    variables = [
        "low_percentile",  # Low percentile threshold of spread. Default 25th percentile
        "high_percentile",  # High percentile threshold of spread. Default 75th percentile
        "min_volume",  # Mininum trade volume of a bond to be selected. Default 1 billion
        "lookback_years" # Period of historical data to be used for analysis. Default 3 years
    ]

    def __init__(    
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ):
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        self.low_percentile = setting.get("low_percentile", 25) 
        self.high_percentile = setting.get("high_percentile", 75) 
        self.min_volume = setting.get("min_volume", 1e9) 
        self.lookback_years = setting.get("lookback_years", 3) 

    def add_coupon(self, code):
        """
        At the end of each day, adds coupon payment (if any) into broker at T+1.
        Coupon payment is based on T+1's actual coupon.
        This method is supposed to be called in next().
        """
        cash_flow = pd.read_csv(
            const.INDEX_ENHANCEMENT.CDB_PATH + f"cash_flow_{code}.csv",
            parse_dates=["DATE"],
        )
        cash_flow["DATE"] = cash_flow["DATE"].dt.date

        # date(0) and date(1) are trade days which may not be consecutive and skip coupon
        # payments between them.
        today_remaining_payment_times = len(
            cash_flow[cash_flow["DATE"] > self.getdatabyname(code).datetime.date(0)]
        )
        tomorrow_remaining_payment_times = len(
            cash_flow[cash_flow["DATE"] > self.getdatabyname(code).datetime.date(1)]
        )
        if tomorrow_remaining_payment_times < today_remaining_payment_times:
            coupon_row_num = len(cash_flow) - today_remaining_payment_times
            # Bond holder will receive coupon payments at T+1.
            coupon = (
                cash_flow.iloc[coupon_row_num]["AMOUNT"]
                * self.getposition(self.getdatabyname(code)).size
            )
            self.log(
                f"coupon payment {code} {coupon}",
                dt=self.getdatabyname(code).datetime.date(1),
            )
            self.broker.add_cash(coupon)



