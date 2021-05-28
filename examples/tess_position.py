import pandas as pd
import datetime
from fxincome.const import COUPON_TYPE
from fxincome.const import CASHFLOW_TYPE
from dateutil.relativedelta import relativedelta
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
import numpy as np
from fxincome.asset import Bond
from fxincome.position import Position_Bond


num=4
if num==0:
    code='200016'
    initial_date=datetime.datetime(2020,11,19)
    end_date=datetime.datetime(2030,11,19)
    issue_price=100
    coupon_rate=3.27
    coupon_type='附息'
    coupon_frequency=2
    bond=Bond(code, initial_date, end_date,issue_price, coupon_rate,coupon_type, coupon_frequency)

    num='001'
    account_type='TPL'
    begin_quantity=10000#元
    begin_date=datetime.datetime(2021,11,18)
    begin_cleanprice=101.2158
    positon_bond=Position_Bond(num,bond,account_type,begin_quantity,begin_date,begin_cleanprice)


    # bond.get_dailycoupon(datetime.datetime(2030,11,19))

    positon_bond.move(datetime.datetime(2021,11,23),3.12,-10000)
    positon_bond.move(datetime.datetime(2021,11,25),3.14)
    positon_bond.move(datetime.datetime(2021,11,28),3.14)
    print(positon_bond.position_gain_df)
    print(positon_bond.get_cashflow('All'))



    # print(positon_bond.begin_dirtyprice)
    # print(positon_bond.begin_ytm)
    #
    # print(positon_bond.get_cashflow('History'))
    # print(positon_bond.get_cashflow('Undelivered'))
    # print(positon_bond.get_cashflow('All'))
if num==1:
    code='120015'
    initial_date=datetime.datetime(2012,8,23)
    end_date=datetime.datetime(2022,8,23)
    issue_price=100
    coupon_rate=3.39
    coupon_type='附息'
    coupon_frequency=2
    bond=Bond(code, initial_date, end_date,issue_price, coupon_rate,coupon_type, coupon_frequency)

    num='002'
    RealDailyR=0.00743367463484038#有百分号的
    AmortizedPrice=101.5368322812683

    account_type='OCI'
    begin_quantity=10000#元
    begin_date=datetime.datetime(2022,2,20)
    begin_cleanprice=101.5368322812683

    positon_bond=Position_Bond(num,bond,account_type,begin_quantity,begin_date,begin_cleanprice)
    # positon_bond.move(datetime.datetime(2022,8,23),3.12)
    # positon_bond.move_ytm(datetime.datetime(2022,2,21),3.14,-5000)
    positon_bond.move_ytm(datetime.datetime(2022,2,22),3.14,-5000)
    positon_bond.move_ytm(datetime.datetime(2022,7,23),3.1,-5000)
    positon_bond.move_ytm(datetime.datetime(2022,8,22),3.14)
    positon_bond.move_ytm(datetime.datetime(2022,8,25))

    print(positon_bond.position_gain_df)
    print(positon_bond.get_cashflow('All'))
    print(positon_bond.get_cashflow('All')['cash'].sum())
    print(positon_bond.date)
    print(positon_bond.quantity)
if num==2:
    code='120015'
    initial_date=datetime.datetime(2012,8,23)
    end_date=datetime.datetime(2022,8,23)
    issue_price=100
    coupon_rate=3.39
    coupon_type='附息'
    coupon_frequency=2
    bond=Bond(code, initial_date, end_date,issue_price, coupon_rate,coupon_type, coupon_frequency)

    date=datetime.datetime(2022,8,23)+datetime.timedelta(days=-31)
    print(date)
    curve_df=pd.DataFrame([[0, 2], [30, 2.5], [90, 2.8], [365, 3.0],[730,3.2],[822,3.21],[1095,3.5]], columns=['days', 'rate'])

    print(curve_df)
    print(isinstance(curve_df,pd.core.frame.DataFrame))

    print(bond.curve_to_ytm(date, curve_df))
    print(bond.curve_to_dirtyprice(date, curve_df))
    print(bond.curve_to_cleanprice(date, curve_df))
if num==3:
    code='120015'
    initial_date=datetime.datetime(2012,8,23)
    end_date=datetime.datetime(2022,8,23)
    issue_price=100
    coupon_rate=3.39
    coupon_type='附息'
    coupon_frequency=2
    bond=Bond(code, initial_date, end_date,issue_price, coupon_rate,coupon_type, coupon_frequency)

    num='002'
    RealDailyR=0.00743367463484038#有百分号的
    AmortizedPrice=101.5368322812683

    account_type='TPL'
    begin_quantity=10000#元
    begin_date=datetime.datetime(2022,2,21)
    begin_cleanprice=101.5368322812683



    positon_bond=Position_Bond(num,bond,account_type,begin_quantity,begin_date,begin_cleanprice)
    # positon_bond.move(datetime.datetime(2022,8,23),3.12)
    curve_df=pd.DataFrame([[0, 2], [30, 2.5], [90, 2.8], [365, 3.0],[730,3.2],[822,3.21],[1095,3.5]], columns=['days', 'rate'])
    positon_bond.move_curve(datetime.datetime(2022,3,1),curve_df,-5000)
    # positon_bond.move_curve(datetime.datetime(2022,5,23),curve_df,-2000)
    # positon_bond.move_curve(datetime.datetime(2022,8,24))

    print(positon_bond.position_gain_df)
    print(positon_bond.get_cashflow('All'))
    print(positon_bond.get_cashflow('All')['cash'].sum())
    print(positon_bond.date)
    print(positon_bond.quantity)
if num==4:
    code='120015'
    initial_date=datetime.datetime(2012,8,23)
    end_date=datetime.datetime(2022,8,23)
    issue_price=100
    coupon_rate=3.39
    coupon_type='附息'
    coupon_frequency=2
    bond=Bond(code, initial_date, end_date,issue_price, coupon_rate,coupon_type, coupon_frequency)
    date=datetime.datetime(2021,5,17)
    ytm=3.0536986301369864
    curve_df=pd.DataFrame([[0, 2], [30, 2.5], [90, 2.8], [365, 3.0],[730,3.2],[822,3.21],[1095,3.5]], columns=['days', 'rate'])
    DURARION_TYPE='Macaulay'
    DURARION_TYPE='Modified'
    print(bond.ytm_to_dv01(date, ytm))
    print(bond.curve_to_dv01(date, curve_df))
    print(bond.ytm_to_duration(date, ytm, DURARION_TYPE))
    print(bond.curve_to_duration(date, curve_df, DURARION_TYPE))

    RealDailyR = 0.00743367463484038  # 有百分号的
    AmortizedPrice = 101.5368322812683
    account_type = 'TPL'
    begin_quantity = 10000  # 元
    begin_date = datetime.datetime(2022, 2, 21)
    begin_cleanprice = 101.5368322812683
    positon_bond=Position_Bond(num,bond,account_type,begin_quantity,begin_date,begin_cleanprice)

    positon_bond.move_curve(datetime.datetime(2022, 2, 28), curve_df,-5000)
    positon_bond.move_curve(datetime.datetime(2022,3,1),curve_df,-5000)
    positon_bond.move_curve(datetime.datetime(2023, 3,2), curve_df)
    print(positon_bond.position_gain_df)
    print(positon_bond.get_cashflow('All'))
    print(positon_bond.get_cashflow('All')['cash'].sum())
    print(positon_bond.date)
    print(positon_bond.quantity)

