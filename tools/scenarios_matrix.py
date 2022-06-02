import copy
import sys
import time
from tqdm import tqdm
import pandas as pd
import datetime
from fxincome.asset import Bond
from fxincome.position import Position_Bond

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

address = r'.\scenarios_matrix.xlsx'

parameter = pd.read_excel(address, sheet_name='parameter', index_col=False).set_index('参数')
base_date = parameter.at['基准日', '数值']
pips_all = [int(i) for i in parameter.at['基点变化值', '数值'].split(',')]
days_all = [int(i) for i in parameter.at['日期变化值', '数值'].split(',')]
reinvest_rate = float(parameter.at['再投资收益率', '数值'])

bond = pd.read_excel(address, sheet_name='holding')
bond['end_date'] = pd.to_datetime(bond['end_date'])
bond['initial_date'] = pd.to_datetime(bond['initial_date'])

date_all = [base_date + datetime.timedelta(i) for i in days_all]
position_list = []
for index, row in bond.iterrows():
    asset_i = Bond(row['code'],
                   row['initial_date'],
                   row['end_date'],
                   row['issue_price'],
                   row['coupon_rate'],
                   row['coupon_type'],
                   row['coupon_frequency'])
    cleanprice = asset_i.ytm_to_cleanprice(base_date, row['ytm'])
    position_i = Position_Bond(index,
                               asset_i,
                               'TPL',
                               row['holding'],
                               base_date,
                               cleanprice)
    position_list.append(position_i)
result_list = []
all_size = len(pips_all) * len(position_list) * len(date_all)
with tqdm(total=all_size) as step:
    for pips in pips_all:  # [-30,-20,-10,0,10,20,30]:
        position_list_i = copy.deepcopy(position_list)
        for position_i in position_list_i:
            ytm = position_i.begin_ytm + pips / 100
            for newdate in date_all:
                position_i.move_ytm(newdate, ytm)
                step.update(1)
            position_i.reinvest(reinvest_rate)
            show = position_i.get_position_gain()
            show = show[show['date'].isin(date_all)].copy()
            if show.empty == False:
                show['code'] = position_i.bond.code
                show['pips'] = pips
                show['maturity'] = round((position_i.bond.end_date - base_date).days / 365, 2)
                show['value'] = position_i.begin_quantity * position_i.begin_dirtyprice / 100
                result_list.append(show[['code', 'date', 'pips', 'gain_sum', 'maturity', 'value', 'reinvest_interest',
                                         'interest_sum', 'price_gain_sum', 'float_gain_sum']])
result_df = pd.concat(result_list, ignore_index=True)
result_df['days'] = result_df['date'].apply(lambda x: 'T+{}'.format((x - base_date).days))
agg_df = result_df.groupby(['pips', 'date']) \
    .apply(lambda x: pd.Series({
    'IRR': x['gain_sum'].sum() / x['value'].sum(),
    'maturity': round(sum(x['maturity'] * x['value']) / x['value'].sum(), 2)
})).reset_index()
agg_df['IRR'] = agg_df.apply(lambda x: round(x['IRR'] / (x['date'] - base_date).days * 365 * 100, 2), axis=1)
# print(agg_df)
pivot_df = pd.pivot_table(agg_df, index='pips', columns='date', values='IRR')
pivot_df.columns = ['T+{}'.format((i - base_date).days) for i in pivot_df.columns]

agg_df = result_df.groupby(['pips', 'date']).\
    apply(lambda x: pd.Series({
    'gain_sum': x['gain_sum'].sum(),
    'maturity': round(sum(x['maturity'] * x['value']) / x['value'].sum(), 2)
})).reset_index()
gain_df = pd.pivot_table(agg_df, index='pips', columns='date', values='gain_sum')
gain_df.columns = ['T+{}'.format((i - base_date).days) for i in gain_df.columns]

time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
address = r'.\result\ss_result_{}.xlsx'.format(time)
wirter = pd.ExcelWriter(address)
parameter = parameter.reset_index()[['参数', '数值']]
parameter = parameter.iloc[:4, :]

result_df['reinvest_interest'] = result_df['reinvest_interest'].fillna(0)
result_df = result_df[
    ['code', 'pips', 'days', 'gain_sum', 'interest_sum', 'float_gain_sum', 'price_gain_sum', 'reinvest_interest']]
parameter.to_excel(wirter, sheet_name='parameter', index=False)
bond.to_excel(wirter, sheet_name='holding', index=False)
pivot_df.to_excel(wirter, sheet_name='result_rate')
gain_df.to_excel(wirter, sheet_name='result_value')
result_df.to_excel(wirter, sheet_name='detail')
wirter.save()
