from fxincome.asset import Bond
from fxincome.utils import get_curve
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas.api.types import CategoricalDtype
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)


def maxx(x, i):
    i = len(x) if len(x) < i else i
    sort_x = sorted(x)[-i]
    return sort_x


def parse_dates(periods, today):
    """
        把“1D, 3M, 2Y” 这种字符串变成一个2D list。list中每个元素的第2个为end_date，end_date = today + 相应的时间。
        list的样子是： [['1D', 'end_date_1'], ['3M', 'end_date_2'], ['2Y', 'end_date_3']]
        Args:
            periods(str): 样子形如“1D, 3M, 2Y”
            today(datetime): 基准日期
        Returns:
            dates(list): [['1D', 'end_date_1'], ['3M', 'end_date_2'], ['2Y', 'end_date_3']]，按照日期从小到大排序
                        其中end_date类型为datetime
    """
    periods = periods.replace(' ', '').split(',')
    dates = []
    for k in periods:
        delta = int(k[:-1])
        if k[-1] == 'D':
            end_date_k = today + relativedelta(days=delta)
        elif k[-1] == 'M':
            end_date_k = today + relativedelta(months=delta)
        elif k[-1] == 'Y':
            end_date_k = today + relativedelta(years=delta)
        else:
            raise NotImplementedError("Wrong period input. Only 'D', 'M', 'Y' allowed.")
        dates.append([k, end_date_k])
    dates = np.array(dates)
    dates = dates[dates[:, 1].argsort()]
    return dates.tolist()


if __name__ == '__main__':
    address = './rolldown_compare.xlsx'
    asset_df = pd.read_excel(address, header=3, sheet_name='asset')
    parameter_df = pd.read_excel(address, sheet_name='parameter').set_index('参数')
    date = parameter_df.at['基准日', '数值']
    bond_type_need = ['政策银行债', '国债', '地方政府债']
    min_max = parameter_df.at['收益率曲线最短最长期限', '数值']
    min_max_list = parse_dates(min_max, date)
    min_date = min_max_list[0][1]
    max_date = min_max_list[1][1]
    asset_df['initial_date'] = pd.to_datetime(asset_df['initial_date'])
    asset_df['end_date'] = pd.to_datetime(asset_df['end_date'])
    #  按照券种、交易量排名等筛选债券
    asset_df = asset_df[(asset_df['bond_type'].isin(bond_type_need)) &
                        (asset_df['code'].str.contains('IB'))]
    asset_df['period'] = asset_df['end_date'].apply(lambda x: round((x - date).days / 365))
    asset_df['period2'] = asset_df['end_date'].apply(lambda x: round((x - date).days / 365, 2))
    #  每个关键期限按交易量大小选出2只代表券，用于描绘收益率曲线
    asset_df['ranking'] = asset_df[['trading', 'period']].groupby('period').transform(lambda x: x >= maxx(x, 2))
    asset_df = asset_df[(asset_df['ranking']) & (asset_df['trading'] > 0)].sort_values(['period2'], ignore_index=True)
    yield_df = asset_df.iloc[:, 10:]
    #  构造收益率曲线
    curve_dot = yield_df[['period2', 'ytm']].to_numpy()
    curve = get_curve(curve_dot, 'HERMIT')
    #  按照日期区间筛选展示债券
    asset_df = asset_df[(asset_df['end_date'] >= min_date) &
                        (asset_df['end_date'] <= max_date)]
    hold_time = parameter_df.at['持有期限', '数值']
    period_list = parse_dates(hold_time, date)
    asset_dic = {}
    for i, j in asset_df.iterrows():
        bond_i = Bond(code=j['code'],
                      initial_date=j['initial_date'],
                      end_date=j['end_date'],
                      issue_price=j['issue_price'],
                      coupon_rate=j['coupon_rate'],
                      coupon_type=j['coupon_type'],
                      coupon_frequency=j['coupon_frequency'])
        asset_dic[j['code']] = bond_i

        for k in period_list:
            if k[1] <= j['end_date']:
                end_ytm = curve((j['end_date'] - k[1]).days / 365)
                asset_df.loc[i, k[0]] = bond_i.get_profit(date, k[1], j['ytm'], end_ytm)[1]

    asset_df['bond'] = asset_df.apply(lambda x: '{}[{}Y][{:.2f}%]'.format(x['code'], x['period2'], x['ytm']), axis=1)
    hold_time_title = parameter_df.at['持有期限', '数值'].split(',')
    reuslt_overview = asset_df[['bond'] + hold_time_title].set_index('bond')
    print(reuslt_overview)
    reuslt_overview = reuslt_overview.applymap(lambda x: round(x, 2) if pd.notnull(x) else x)

    asset_df = asset_df.set_index('code')
    result_dic = {}

    for k in period_list:
        # print(k[0])
        column_k = asset_df[asset_df['end_date'] >= k[1]].index
        column_k = [[i, j] for i in column_k for j in column_k]
        reuslt_k = pd.DataFrame(column_k, columns=['bond_base', 'bond_target'])
        reuslt_k['bond_base_ytm'] = reuslt_k['bond_base'].apply(lambda x: asset_df.loc[x, 'ytm'])
        reuslt_k['bond_base_yeild'] = reuslt_k['bond_base'].apply(lambda x: asset_df.loc[x, k[0]])
        reuslt_k['bond_base_period'] = reuslt_k['bond_base'].apply(lambda x: asset_df.loc[x, 'period2'])

        reuslt_k['bond_target_ytm'] = reuslt_k['bond_target'].apply(lambda x: asset_df.loc[x, 'ytm'])
        reuslt_k['bond_target_yeild'] = reuslt_k['bond_target'].apply(lambda x: asset_df.loc[x, k[0]])
        reuslt_k['bond_target_period'] = reuslt_k['bond_target'].apply(lambda x: asset_df.loc[x, 'period2'])
        total_k = len(reuslt_k)
        print(reuslt_k)

        for i, j in reuslt_k.iterrows():
            print('{}:{:.0f}/{:.0f},{:.2%}'.format(k[0], i, total_k, (i + 1) / total_k))

            if j['bond_base'] == j['bond_target']:
                # base_bp=(curve((asset_df.loc[j['bond_base'],'end_date']-k[1]).days/365)-j['bond_base_ytm'])*100
                reuslt_k.loc[i, 'bp'] = 0
            else:
                y1 = -50
                y2 = 50
                y_last = 0
                while True:
                    y = (y1 + y2) / 2
                    # if i==18:
                    #     print(y)
                    yld = asset_dic[j['bond_base']].get_profit(date, k[1], j['bond_base_ytm'], y)[1]
                    if abs(yld - j['bond_target_yeild']) < 0.01:
                        break
                    if (abs(yld - j['bond_target_yeild']) < 0.1) & (abs(y - y_last) < 0.0001):
                        break
                    if yld < j['bond_target_yeild']:
                        y2 = y
                    else:
                        y1 = y
                    # if i==18:
                    #     print(y,y_last,yeild,j['bond_target_yeild'])

                    y_last = y
                reuslt_k.loc[i, 'bp'] = (y - curve((asset_df.loc[j['bond_base'], 'end_date'] - k[1]).days / 365)) * 100

        reuslt_k['base_bond'] = reuslt_k.apply(
            lambda x: '{}\n[{}Y]\n[{:.2f}%]\n[{:.2f}%]'.format(x['bond_base'], x['bond_base_period'],
                                                               x['bond_base_ytm'], x['bond_base_yeild']), axis=1)
        reuslt_k['target_bond'] = reuslt_k.apply(
            lambda x: '{}\n[{}Y]\n[{:.2f}%]\n[{:.2f}%]'.format(x['bond_target'], x['bond_target_period'],
                                                               x['bond_target_ytm'], x['bond_target_yeild']), axis=1)
        rank_type = CategoricalDtype(list(reuslt_k['base_bond'].drop_duplicates()), ordered=True)

        reuslt_k['base_bond'] = reuslt_k['base_bond'].astype(rank_type)
        reuslt_k['target_bond'] = reuslt_k['target_bond'].astype(rank_type)
        result_k_pivot = pd.pivot_table(reuslt_k, index='base_bond', columns='target_bond', values='bp',
                                        aggfunc='sum', margins=True, margins_name='sum')
        result_k_pivot = result_k_pivot.round(2)
        result_dic[k[0]] = result_k_pivot

    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    address = r'.\result\rc_result_{}.xlsx'.format(time)
    writer = pd.ExcelWriter(address)
    reuslt_overview.to_excel(writer, sheet_name='result')
    for i, j in result_dic.items():
        j.to_excel(writer, sheet_name=i)
    writer.save()

    #  描绘收益率曲线。为了只展示指定期限范围内的收益率曲线，重新筛选取样点
    yield_df = asset_df.iloc[:, 10:]
    curve_dot = yield_df[['period2', 'ytm']].to_numpy()
    curve = get_curve(curve_dot, 'HERMIT')
    plt.figure()
    x = np.linspace(curve_dot[0, 0], curve_dot[-1, 0], 10000)
    plt.plot(x, [curve(i) for i in x])
    plt.scatter(curve_dot[:, 0], curve_dot[:, 1], marker='*')
    plt.xticks(range(int(curve_dot[0, 0]), int(curve_dot[-1, 0]) + 2))
    plt.grid(True)

    address = r'.\result\rc_result_{}.jpg'.format(time)
    plt.savefig(address, dpi=600)
