import pandas as pd
from fxincome.asset import Bond
import datetime

from fxincome.position import Position_Bond
from fxincome.portfolio import Portfolio_Bond
address=r'C:\Users\zyzse\Desktop\try.xlsx'
def excel_to_portfolio_bond(address,fill_curve=True):
    asset_df=pd.read_excel(address,sheet_name='asset')
    # print(asset_df)
    asset_dic={}
    for index,row in asset_df.iterrows():
        asset_i=Bond(row['code'],
                     datetime.datetime.strptime(row['initial_date'],'%Y-%m-%d'),
                     datetime.datetime.strptime(row['end_date'],'%Y-%m-%d'),
                     row['issue_price'],
                     row['coupon_rate'],
                     row['coupon_type'],
                     row['coupon_frequency'])
        asset_dic[row['code']]=asset_i

    position_df=pd.read_excel(address,sheet_name='position')
    # print(position_df)
    position_dic={}
    for index,row in position_df.iterrows():
        position_i=Position_Bond(row['id'],
                                 asset_dic[row['code']],
                                 row['account_type'],
                                 row['begin_quantity'],
                                 row['begin_date'],
                                 row['begin_cleanprice'])
        position_dic[row['id']]=position_i
    curve_df=pd.read_excel(address,sheet_name='curve')
    curve_df=curve_df.iloc[:,1:]
    init_curve_date_list=list(curve_df.columns)
    # print(curve_df)
    if fill_curve:
        begin_date=curve_df.columns[1]
        end_date=curve_df.columns[-1]
        curve_all_df=pd.DataFrame(pd.date_range(begin_date,end_date,freq='D'),columns=['date'])
        curve_df=curve_df.set_index('days').T.reset_index()
        curve_df.rename(columns={'index':'date'},inplace=True)
        curve_all_df=pd.merge(curve_all_df,curve_df,how='left')
        curve_all_df=curve_all_df.interpolate().set_index('date').T.reset_index()
        curve_df=curve_all_df.rename(columns={'index':'days'})



    trade_df=pd.read_excel(address,sheet_name='trade')
    trade_df['date']=pd.to_datetime(trade_df['date'])
    # print(trade_df)


    porfolio=Portfolio_Bond(asset_dic,position_dic,curve_df,trade_df,init_curve_date_list)
    porfolio.cal_result()
    return porfolio