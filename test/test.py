import pandas as pd

from fxincome.const import COUPON_TYPE
from fxincome.const import CASHFLOW_TYPE
from dateutil.relativedelta import relativedelta
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
from fxincome.asset import Bond
from fxincome.position import Position_Bond
from fxincome.portfolio import Portfolio_Bond
from fxincome.reading import excel_to_portfolio_bond

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


address=r'C:\Users\A\Desktop\try.xlsx'
portfolio_bond=excel_to_portfolio_bond(address)
# print(portfolio_bond.get_cashflow())
# print(portfolio_bond.get_cashflow('Agg'))
# print(portfolio_bond.get_position_gain('Raw'))
# print(portfolio_bond.get_position_gain('Agg'))
# print(portfolio_bond.get_view())
portfolio_bond.get_plot()
