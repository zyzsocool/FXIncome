import pandas as pd
from fxincome.asset import Bond
from fxincome.portfolio import Portfolio_Bond
import datetime
address_in = r'.\classical_situation.xlsx'
time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
address_out = r'.\result\cs_result_{}.xlsx'.format(time)

data=pd.read_excel(address_in, sheet_name=None)

parameter=data['parameter'].set_index('参数')
initial_cash=parameter.at['initial_cash','数值']
repo_rate=parameter.at['repo_rate','数值']


asset_df=data['asset']
asset_dic={}
asset_df['initial_date']=pd.to_datetime(asset_df['initial_date'])
asset_df['end_date']=pd.to_datetime(asset_df['end_date'])
for index, row in asset_df.iterrows():
    asset_i = Bond(row['code'],
                   row['initial_date'],
                   row['end_date'],
                   row['issue_price'],
                   row['coupon_rate'],
                   row['coupon_type'],
                   row['coupon_frequency'],
                   bond_type=row['bond_type'])
    asset_dic[row['code']]=asset_i
trad_df=data['trade']
trad_df['date']=pd.to_datetime(trad_df['date'])
curve_df=data['curve']


porfolio_test=Portfolio_Bond(asset_dic)
porfolio_test.move_onestep(trad_df,curve_df,initial_cash=initial_cash,repo_rate=repo_rate)
cashflow_raw_df,cashflow_agg_df=porfolio_test.get_cashflow()
position_gain_raw_df, position_gain_agg_l1_df, position_gain_agg_l2_df=porfolio_test.get_position_gain()

display_date=list(set(position_gain_raw_df[pd.isnull(position_gain_raw_df['market_cleanprice'])]['date']))
position_gain_agg_l1_df=position_gain_agg_l1_df[position_gain_agg_l1_df['date'].isin(display_date)==False].reset_index(drop=True)
position_gain_agg_l2_df=position_gain_agg_l2_df[position_gain_agg_l2_df['date'].isin(display_date)==False].reset_index(drop=True)

wirter = pd.ExcelWriter(address_out)
data['parameter'].to_excel(wirter, sheet_name='parameter', index=False)
data['asset'].to_excel(wirter, sheet_name='asset', index=False)
data['trade'].to_excel(wirter, sheet_name='trade', index=False)
data['curve'].to_excel(wirter, sheet_name='curve', index=False)
position_gain_agg_l2_df.to_excel(wirter, sheet_name='position_gain_agg_l2', index=False)
position_gain_agg_l1_df.to_excel(wirter, sheet_name='position_gain_agg_l1', index=False)
cashflow_agg_df.to_excel(wirter, sheet_name='cashflow_agg', index=False)
position_gain_raw_df.to_excel(wirter, sheet_name='position_gain_raw', index=False)
cashflow_raw_df.to_excel(wirter, sheet_name='cashflow_raw', index=False)
wirter.save()