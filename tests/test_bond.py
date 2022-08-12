from financepy.utils import *
from financepy.products.bonds.bond import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


class TestBond:

    @pytest.fixture(scope='class')
    def global_data(self):
        #  210210 is a 10 year CDB bond, 1 coupon payment per year.
        bond = Bond(
            issue_date=Date(7, 6, 2021),
            maturity_date=Date(7, 6, 2031),
            coupon=0.0341,
            freq_type=FrequencyTypes.ANNUAL,
            accrual_type=DayCountTypes.ACT_ACT_ISDA
        )
        # 229936 is a 3 months treasure with 0 coupon per year.
        bill = Bond(
            issue_date=Date(25, 7, 2022),
            maturity_date=Date(24, 10, 2022),
            coupon=0,
            freq_type=FrequencyTypes.ANNUAL,
            accrual_type=DayCountTypes.ACT_365F
        )
        curve_df = pd.DataFrame([[0, 2], [30, 2.5], [90, 2.8], [365, 3.0], [730, 3.2], [822, 3.21], [1095, 3.5]],
                                columns=['days', 'rate'])
        settlement_date = Date(8, 8, 2022)
        return {'bond': bond,
                'bill': bill,
                'curve': curve_df,
                'date': settlement_date}

    def test_yield_to_maturity_coupon(self, global_data):
        bond = global_data['bond']
        assess_date = global_data['date']
        clean_price = 103.0155
        assert bond.yield_to_maturity(assess_date, clean_price, YTMCalcType.US_STREET) * 100 == \
               pytest.approx(3.0150)

    def test_yield_to_maturity_no_coupon(self, global_data):
        bond = global_data['bill']
        assess_date = global_data['date']
        clean_price = 99.7056
        assert bond.yield_to_maturity(assess_date, clean_price, YTMCalcType.US_STREET) * 100 == \
               pytest.approx(1.3998)
