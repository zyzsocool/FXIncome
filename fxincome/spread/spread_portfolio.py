from financepy.utils import *
from financepy.products.bonds import *
import numpy as np
import pandas as pd
import datetime


class SpreadPortfolio:
    """
    The spread strategy is:
    On start date, we borrow a bond(bond_short) and sell it. At the same time, we spend the cash received to buy
    another bond(bond_long). We expect the ytm spread of two bonds will increase or decrease.
    On end date, we sell the bond_long. We spend the cash received to buy the bond_short and return it to the lender.
    If the ytm spread increases or decreases as we expected, we should get some profit at the end.
    Cash flows during the period are considered. Extra cash generates profit at a risk-free rate(rf), and short of cash
    decreases profit at the same risk-free rate(rf).
    All yields or risk-free p/l are calculated assuming 365 days per year.
    Yields' units are exact float numbers, eg. 0.05 for 5.0%.
    """

    @staticmethod
    def profit(bond_short: Bond,
               bond_long: Bond,
               ytm_open_short: float,
               ytm_open_long: float,
               ytm_close_short: float,
               ytm_close_long: float,
               lend_rate: float,
               start_date: Date,
               end_date: Date,
               rf: float = 0.0):
        """
        Core function to calculate strategy profit and yield. It's designed to deal with different scenarios.
        Args:
            bond_short(Bond): A FinancePy Bond to short(expects its ytm to increase).
                              This bond's face_amount decides the scale of profit.
            bond_long(Bond): A FinancePy Bond to long(expects its ytm to decrease).
                             This bond's face_amount decides the scale of profit.
            ytm_open_short(float): Ytm to open a short position of a Bond (Ytm to sell the bond).
            ytm_open_long(float): Ytm to open a long position of a Bond (Ytm to buy the bond).
            ytm_close_short(float): Ytm to close a short position of a Bond (Ytm to buy back the bond).
            ytm_close_long(float): Ytm to close a long position of a Bond (Ytm to sell the bond).
            rf: If set to zero, cash flows during the period will not affect the profit.
        Returns:
            profit(float): profit or loss of the portfolio at the end.
            profit_yield(float): profit / bond_long._face_amount * 365 / (end_date - start_date)
        """

        hold_time = (end_date - start_date) / 365
        price_recv_open_short = bond_short.full_price_from_ytm(start_date, ytm_open_short, YTMCalcType.CFETS)
        price_paid_open_long = bond_long.full_price_from_ytm(start_date, ytm_open_long, YTMCalcType.CFETS)
        coupon_bond_short = []
        coupon_bond_long = []
        for i in range(0, len(bond_short._coupon_dates)):
            # Coupon on the first day is paid to the bondholder on the previous day.
            if start_date < bond_short._coupon_dates[i] <= end_date:
                coupon_bond_short.append(
                    bond_short._coupon / bond_short._frequency * bond_short._face_amount * (
                            1 + rf * (end_date - bond_short._coupon_dates[i]) / 365))
        for i in range(0, len(bond_long._coupon_dates)):
            # Coupon on the first day is paid to the bondholder on the previous day.
            if start_date < bond_long._coupon_dates[i] <= end_date:
                coupon_bond_long.append(
                    bond_long._coupon / bond_long._frequency * bond_long._face_amount * (
                            1 + rf * (end_date - bond_long._coupon_dates[i]) / 365))
        price_paid_close_short = bond_short.full_price_from_ytm(end_date, ytm_close_short, YTMCalcType.CFETS)
        price_recv_close_long = bond_long.full_price_from_ytm(end_date, ytm_close_long, YTMCalcType.CFETS)
        diff_begin = price_recv_open_short * bond_short._face_amount / 100 - price_paid_open_long * bond_long._face_amount / 100
        diff_end = price_recv_close_long * bond_long._face_amount / 100 - price_paid_close_short * bond_short._face_amount / 100
        expense_lend = lend_rate * bond_short._face_amount * hold_time
        profit = diff_end + diff_begin * (1 + rf * hold_time) - expense_lend + sum(coupon_bond_long) - sum(
            coupon_bond_short)
        profit_yield = profit / bond_long._face_amount / hold_time
        return profit, profit_yield

    @staticmethod
    def profit_reach_time(bond_short: Bond,
                          bond_long: Bond,
                          ytm_open_short: float,
                          ytm_open_long: float,
                          lend_rate: float,
                          spread_predict: float,
                          profit_required: float,
                          start_date: Date,
                          ytm_close_short: float = 0.0,
                          ytm_close_long: float = 0.0,
                          max_days: int = 180,
                          rf: float = 0.0,
                          ):
        """
        Define spread = ytm_long - ytm_short. Spreads can be positive or negative.
        Given the predicted spread at the final(assuming the change of spread is good to profit) and initial ytms,
        calculate the MAXIMUM DAYS WE CAN WAIT UNTIL THE PREDICTED SPREAD IS REACHED and REQUIRED PROFIT IS EARNED.
        If any of ytm_close_short and ytm_close_long are missing (default as 0),
        the predicted spread will be used to calculate the ytm_close_short and ytm_close_long, which increases/decreases
        by half of the change of spread.
        If both of ytm_close_short and ytm_close_long are given (not 0),
        those values will be used to calculate, and the spread_predict will be ignored.
        If profit is always < profit_required, last day's profit is closest_profit and max_days is days_required.
        Args:
            bond_short(Bond): A FinancePy Bond to short(expects its ytm to increase).
                              This bond's face_amount decides the scale of profit.
            bond_long(Bond): A FinancePy Bond to long(expects its ytm to decrease).
                             This bond's face_amount decides the scale of profit.
            ytm_open_short(float): Ytm to open a short position of a Bond (Ytm to sell the bond).
            ytm_open_long(float): Ytm to open a long position of a Bond (Ytm to buy the bond).
            spread_predict(float): If any of ytm_close_short and ytm_close_long are missing (default to zero),
                            this spread will be considered the final spread.
                            Assume the change of spread is good to profit.
            profit_required(float): If required profit is not reached during max days, None will be returned as result.
            ytm_close_short(float): Default as 0. If not given, spread_predict will be used.
            ytm_close_long(float): Default as 0. If not given, spread_predict will be used.
            max_days: Maximum days we can wait. max_days = last day - start day.
            rf: If set to zero, cash flows during the period will not affect the profit.
        Returns:
            closest_profit(float): The profit on the end day when the position is closed.
            days_required(float): Maximum days we can wait until the predicted spread is reached and required profit
                                is earned. Given a spread, the longer we wait, the fewer profit will be earned.
        """
        if ytm_close_short == 0.0 or ytm_close_long == 0.0:
            ytm_close_short = (ytm_open_short + ytm_open_long) / 2 - spread_predict / 20000
            ytm_close_long = (ytm_open_short + ytm_open_long) / 2 + spread_predict / 20000
        closest_profit = None
        days_required = None
        prev_profit = 0
        for i in range(1, max_days + 1):
            end_date = start_date.add_days(i)
            profit, _ = SpreadPortfolio.profit(bond_short, bond_long, ytm_open_short, ytm_open_long, ytm_close_short,
                                               ytm_close_long, lend_rate, start_date, end_date, rf)
            if prev_profit >= profit_required >= profit:
                end_date = start_date.add_days(i - 1)
                closest_profit, _ = SpreadPortfolio.profit(bond_short, bond_long, ytm_open_short, ytm_open_long,
                                                           ytm_close_short, ytm_close_long, lend_rate, start_date,
                                                           end_date, rf)
                days_required = i - 1
                break
            # If profit is always < profit_required, last day's profit is closest_profit and max_days is days_required.
            else:
                closest_profit = profit
                days_required = max_days

        return closest_profit, days_required

    @staticmethod
    def profit_reach_bp(bond_short: Bond,
                        bond_long: Bond,
                        ytm_open_short: float,
                        ytm_open_long: float,
                        lend_rate: float,
                        days_after: int,
                        profit_required: float,
                        start_date: Date,
                        rf: float = 0.0,
                        ):
        """
        Define spread = ytm_long - ytm_short. Spreads can be positive or negative.
        Given initial ytm_long, ytm_short, profit_required, calculate the required spread to earn required profit after
        given days.
        Args:
            bond_short(Bond): A FinancePy Bond to short(expects its ytm to increase).
                              This bond's face_amount decides the scale of profit.
            bond_long(Bond): A FinancePy Bond to long(expects its ytm to decrease).
                             This bond's face_amount decides the scale of profit.
            ytm_open_short(float): Ytm to open a short position of a Bond (Ytm to sell the bond).
            ytm_open_long(float): Ytm to open a long position of a Bond (Ytm to buy the bond).
            days_after(int): End day = start day + days_after.
            rf: If set to zero, cash flows during the period will not affect the profit.
        Returns:
            spread(float): The exact final spread to earn the required profit. Its unit is BP.
            Assume the ytm_close_short and ytm_close_long increases/decreases by half of the change of spread.
            ytm_close_short = (ytm_open_short + ytm_open_long) / 2 - spread_predict / 20000
            ytm_close_long = (ytm_open_short + ytm_open_long) / 2 + spread_predict / 20000
        """
        end_date = start_date.add_days(days_after)

        def bp(spread_predict, bond_short, bond_long, ytm_open_short, ytm_open_long, lend_rate, begin, end, profit, rf):
            ytm_close_short = (ytm_open_short + ytm_open_long) / 2 - spread_predict / 20000
            ytm_close_long = (ytm_open_short + ytm_open_long) / 2 + spread_predict / 20000
            return profit - \
                SpreadPortfolio.profit(bond_short, bond_long, ytm_open_short, ytm_open_long, ytm_close_short,
                                       ytm_close_long,
                                       lend_rate, begin, end, rf)[0]

        spread = optimize.newton(
            bp,
            x0=1,
            fprime=None,
            tol=1e-8,
            args=(
                bond_short, bond_long, ytm_open_short, ytm_open_long, lend_rate, start_date,
                end_date,
                profit_required, rf),
            maxiter=50,
            fprime2=None
        )
        return spread
