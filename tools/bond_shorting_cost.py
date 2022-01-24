import datetime
import pandas as pd
from fxincome.asset import Bond
address_in = r'.\bond_shorting_cost.xlsx'
time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
address_out = r'.\result\bsc_result_{}.xlsx'.format(time)

data = pd.read_excel(address_in, sheet_name=None)
parameter=data['parameter'].set_index('参数')
asset_df=data['asset']
asset_df['initial_date']=pd.to_datetime(asset_df['initial_date'])
asset_df['end_date']=pd.to_datetime(asset_df['end_date'])

base_date = parameter.at['基准日', '数值']
days_all = [int(i) for i in parameter.at['日期变化值', '数值'].split(',')]
repo_rate = float(parameter.at['回购利率', '数值'])
def get_net_profit(bond,base_date,base_ytm,target_date,target_ytm,repo_rate):
    base_dirtyprice=bond.ytm_to_dirtyprice(base_date,base_ytm)
    target_dirtyprice=bond.ytm_to_dirtyprice(target_date,target_ytm)
    cashflow=bond.get_cashflow(base_date)
    coupon=cashflow[cashflow['date']<=target_date]['cash'].sum()
    profit=target_dirtyprice-base_dirtyprice+coupon
    repo=-base_dirtyprice*repo_rate*(target_date-base_date).days/365
    net_profit=profit+repo
    return net_profit*1000000
def get_shorting_cost(bond,base_date,base_ytm,target_date,repo_rate):
    target_ytm=base_ytm
    y=get_net_profit(bond,base_date,base_ytm,target_date,target_ytm,repo_rate)
    while abs(y)>0.0001:
        k=(get_net_profit(bond,base_date,base_ytm,target_date,target_ytm+0.001,repo_rate)- \
          get_net_profit(bond,base_date,base_ytm,target_date,target_ytm-0.001,repo_rate))/0.002
        b=y-k*target_ytm
        target_ytm=-b/k
        y=get_net_profit(bond,base_date,base_ytm,target_date,target_ytm,repo_rate)
    return (target_ytm-base_ytm)*100


result=pd.DataFrame([],columns=['code','period(years)']+['{}days'.format(i) for i in  days_all])
for index, row in asset_df.iterrows():
    bond_i = Bond(row['code'],
                  row['initial_date'],
                  row['end_date'],
                  row['issue_price'],
                  row['coupon_rate'],
                  row['coupon_type'],
                  row['coupon_frequency'])
    bp_list=[row['code'],round((row['end_date']-base_date).days/365,2)]
    first_dirtyprice=bond_i.ytm_to_dirtyprice(base_date, row['ytm'])
    for day_i in days_all:
        target_date=base_date+datetime.timedelta(days=day_i)
        bp_i=round(get_shorting_cost(bond_i,base_date,row['ytm'],target_date,repo_rate),2)
        bp_list.append(bp_i)
        get_net_profit(bond_i,base_date,row['ytm'],target_date,row['ytm']+bp_i/100,repo_rate)
    result.loc[index]=bp_list
wirter = pd.ExcelWriter(address_out)
data['parameter'].to_excel(wirter, sheet_name='parameter', index=False)
data['asset'].to_excel(wirter, sheet_name='asset', index=False)
result.to_excel(wirter,sheet_name='result',index=False)
wirter.save()

