from fxincome.asset import Bond
from fxincome.const import *
from datetime import datetime
import pandas as pd
import pytest



class TestBond:

    @pytest.fixture(scope='class')
    def global_data(self):
        # 209901 is a 3 months treasure with 0 coupon per year.
        bond = Bond(code='209901',
                    initial_date=datetime(2020, 1, 6),
                    end_date=datetime(2020, 4, 6),
                    issue_price=99.5320,
                    coupon_rate=1.8911,
                    coupon_type=COUPON_TYPE.ZERO,
                    coupon_frequency=0
                    )
        curve_df = pd.DataFrame([[0, 2], [30, 2.5], [90, 2.8], [365, 3.0], [730, 3.2], [822, 3.21], [1095, 3.5]],
                                columns=['days', 'rate'])
        date = datetime(2020, 1, 10)
        return {'bond': bond,
                'curve': curve_df,
                'date': date}

    def test_curve_to_ytm(self, global_data):
        bond = global_data['bond']
        assess_date = global_data['date']
        curve_df = global_data['curve']
        assert bond.curve_to_ytm(assess_date,curve_df) == pytest.approx(2.785)

    def test_get_cashflow(self, global_data):
        bond = global_data['bond']
        assess_date = global_data['date']
        cf = bond.get_cashflow(assess_date, CASHFLOW_TYPE.Undelivered).reset_index(drop=True)
        target_cf = pd.DataFrame([[datetime(2020, 4, 6), 100.0]], columns=['date', 'cash'])
        assert target_cf.equals(cf)

    def test_ytm_to_dirtyprice(self, global_data):
        bond = global_data['bond']
        assess_date = global_data['date']
        ytm = 1.98
        assert bond.ytm_to_dirtyprice(assess_date, ytm) == pytest.approx(99.5315)

    def test_dirtyprice_to_ytm(self, global_data):
        bond = global_data['bond']
        assess_date = global_data['date']
        dirtyprice = 99.5315
        assert bond.dirtyprice_to_ytm(assess_date, dirtyprice) == pytest.approx(1.98, abs=1e-3)

    def test_ytm_to_cleanprice(self, global_data):
        bond = global_data['bond']
        assess_date = global_data['date']
        ytm = 1.98
        assert bond.ytm_to_cleanprice(assess_date, ytm) == pytest.approx(99.5110)

    def test_accrued_interest(self, global_data):
        bond = global_data['bond']
        assert bond.accrued_interest(global_data['date']) == pytest.approx(0.020571, abs=1e-6)

    def test_ytm_to_dv01(self, global_data):
        bond = global_data['bond']
        assess_date = global_data['date']
        ytm = 1.98
        assert bond.ytm_to_dv01(assess_date, ytm) == pytest.approx(-0.0023, abs=1e-4)

    def test_ytm_to_duration(self, global_data):
        bond = global_data['bond']
        assess_date = global_data['date']
        ytm = 1.98
        dirty_price = bond.ytm_to_dirtyprice(assess_date, ytm)
        duration = bond.ytm_to_duration(assess_date, ytm, 'Modified')
        dv01 = -dirty_price * duration * 0.0001
        assert pytest.approx(dv01, abs=1e-4) == bond.ytm_to_dv01(global_data['date'], ytm)