# coding: utf-8

from financepy.utils import *
from financepy.products.bonds import *
import pandas as pd
import pytest
from fxincome.spread.spread_portfolio import SpreadPortfolio


class TestSpreadPortfolio:

    @pytest.fixture(scope='class')
    def global_data(self):
        # 220208.IB
        bond_short = Bond(
            issue_date=Date(16, 6, 2022),
            maturity_date=Date(16, 6, 2027),
            coupon=2.69 / 100,
            freq_type=FrequencyTypes.ANNUAL,
            accrual_type=DayCountTypes.ACT_ACT_ICMA,
            face_amount=ONE_MILLION * 50)
        # 170215.IB
        bond_long = Bond(
            issue_date=Date(24, 8, 2017),
            maturity_date=Date(24, 8, 2027),
            coupon=4.24 / 100,
            freq_type=FrequencyTypes.ANNUAL,
            accrual_type=DayCountTypes.ACT_ACT_ICMA,
            face_amount=ONE_MILLION * 50)
        df = pd.read_csv('./test_data.csv', parse_dates=['begin', 'end'])
        return {'bond_short': bond_short,
                'bond_long': bond_long,
                'df': df
                }

    def test_profit(self, global_data):
        df = global_data['df']
        df = df[df["function"] == "profit"]
        for row in df.itertuples(index=False):
            profit, profit_yield = SpreadPortfolio.profit(
                bond_short=global_data['bond_short'],
                bond_long=global_data['bond_long'],
                ytm_open_short=row.ytm_open_short,
                ytm_open_long=row.ytm_open_long,
                ytm_close_short=row.ytm_close_short,
                ytm_close_long=row.ytm_close_long,
                lend_rate=row.lend_rate,
                rf=row.rf,
                start_date=Date(row.begin.day, row.begin.month, row.begin.year),
                end_date=Date(row.end.day, row.end.month, row.end.year))
            assert profit == pytest.approx(row.profit)
            assert profit_yield == pytest.approx(row.profit_yield, abs=1e-8)

    def test_profit_reach_time(self, global_data):
        df = global_data['df']
        df = df[df["function"] == "profit_reach_time"]
        for row in df.itertuples(index=False):
            profit, days_required = SpreadPortfolio.profit_reach_time(
                bond_short=global_data['bond_short'],
                bond_long=global_data['bond_long'],
                ytm_open_short=row.ytm_open_short,
                ytm_open_long=row.ytm_open_long,
                lend_rate=row.lend_rate,
                spread_predict=row.spread_predict,
                profit_required=row.profit_input,
                start_date=Date(row.begin.day, row.begin.month, row.begin.year))
            assert profit == pytest.approx(row.profit)
            assert days_required == pytest.approx(row.days_required)

    def test_profit_reach_hp(self, global_data):
        df = global_data['df']
        df = df[df["function"] == "profit_reach_bp"]
        for row in df.itertuples(index=False):
            bp = SpreadPortfolio.profit_reach_bp(
                bond_short=global_data['bond_short'],
                bond_long=global_data['bond_long'],
                ytm_open_short=row.ytm_open_short,
                ytm_open_long=row.ytm_open_long,
                lend_rate=row.lend_rate,
                days_after=row.days_required,
                profit_required=row.profit_input,
                start_date=Date(row.begin.day, row.begin.month, row.begin.year))
            assert bp == pytest.approx(row.spread_predict)
