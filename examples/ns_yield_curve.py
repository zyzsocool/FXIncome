import os.path
import pandas as pd
import matplotlib as plt
import datetime
from financepy.utils import *
from financepy.products.bonds import *

root_path = 'd:/ProjectRicequant/fxincome/'


def to_bond_freq(freq):
    if freq == 1:
        return FrequencyTypes.ANNUAL
    elif freq == 2:
        return FrequencyTypes.SEMI_ANNUAL
    elif freq == 4:
        return FrequencyTypes.QUARTERLY
    else:
        raise NotImplementedError("Unknown frequency")


def plot_bond_ylds(yld_date: datetime, bonds: list, ylds: list):
    plt.figure(figsize=(12, 6))
    title = yld_date.strftime('%Y%m%d')
    plt.title(f'CDB YTM @{title}')
    yld_date = from_datetime(yld_date.date())
    x = []
    for bond in bonds:
        years_to_maturity = (bond._maturity_date - yld_date) / 365
        x.append(years_to_maturity)
    bond_ylds_scaled = [n * 100 for n in ylds]
    plt.plot(x, bond_ylds_scaled, 'o')

    plt.xlabel('Time to Maturity (years)')
    plt.ylabel('Yield To Maturity (%)')
    plt.ylim((min(bond_ylds_scaled) - 0.3, max(bond_ylds_scaled) * 1.1))
    plt.grid(True)
    plt.savefig(f'./results/EXCEPTION_ytm_{title}.jpg', dpi=600)
    plt.close()


def preprocess():
    source = 'cdb_valuation.csv'
    result = 'cdb_valuation_processed.csv'
    df = pd.read_csv(os.path.join(root_path, source), parse_dates=['date', 'issue_date', 'maturity_date'],
                     encoding='gb2312')
    df = df[df.coupon_type == '固定利率']
    df = df[(df.coupon_info == '附息') | (df.coupon_info == '到期一次还本付息')]
    #  Wind的付息次数不正确，到期一次还本付息的付息次数显示为0，应修改为1
    df.loc[df.coupon_info == '到期一次还本付息', 'coupon_freq'] = 1
    df = df[df.outstanding_balance >= 1000]
    df = df[df.bond_code.str.len() == 9]
    df.to_csv(os.path.join(root_path, result), index=False, encoding='gb2312')


def main():
    source = 'cdb_valuation_processed.csv'
    df_src = pd.read_csv(os.path.join(root_path, source), parse_dates=['date', 'issue_date', 'maturity_date'],
                         encoding='gb2312')
    ns_stats = pd.DataFrame(columns=['date', 'beta1', 'beta2', 'beta3', 'tau'])
    start = datetime.datetime(2022, 6, 1)
    end = datetime.datetime(2022, 7, 28)
    ytm_date = start
    while ytm_date <= end:
        df = df_src[df_src.date == ytm_date]
        if len(df) == 0:  # Today is not a buisness day.
            ytm_date += datetime.timedelta(days=1)
            continue
        bonds = []
        ytms = []
        for bond in df.itertuples():
            issue_date = from_datetime(bond.issue_date)
            maturity_date = from_datetime(bond.maturity_date)
            coupon = bond.coupon / 100
            freq = to_bond_freq(bond.coupon_freq)
            accrual_type = DayCountTypes.ACT_ACT_ISDA
            ytm = bond.ytm / 100
            b = Bond(issue_date, maturity_date, coupon, freq, accrual_type)
            bonds.append(b)
            ytms.append(ytm)

        eval_date = from_datetime(ytm_date)
        try:
            curve_function = CurveFitNelsonSiegel()
            fitted_curve = BondYieldCurve(eval_date, bonds, ytms, curve_function)
            fitted_curve.plot('CDB Yield Curve by Nelson Siegel')
            beta1 = curve_function._beta1
            beta2 = curve_function._beta2
            beta3 = curve_function._beta3
            tau = curve_function._tau
            ns_stats.loc[len(ns_stats)] = [ytm_date, beta1, beta2, beta3, tau]
            # bounds = [(0, -1, -1, -1, 0, 1), (1, 1, 1, 1, 5, 5)]
            # curve_function = CurveFitNelsonSiegelSvensson(bounds=bounds)
            # fitted_curve = BondYieldCurve(eval_date, bonds, ytms, curve_function)
            # fitted_curve.plot('CDB Yield Curve by Nelson Siegel Svensson')
            # curve_function = CurveFitPolynomial()
            # fitted_curve = BondYieldCurve(eval_date, bonds, ytms, curve_function)
            # fitted_curve.plot('CDB Yield Curve by Polynomial')
            # curve_function = CurveFitPolynomial(5)
            # fitted_curve = BondYieldCurve(eval_date, bonds, ytms, curve_function)
            # fitted_curve.plot('CDB Yield Curve by Quintic Polynomial')
            # curve_function = CurveFitBSpline()
            # fitted_curve = BondYieldCurve(eval_date, bonds, ytms, curve_function)
            # fitted_curve.plot('CDB Yield Curve by B Spline')
            # curve_function = CurveFitBSpline(4)
            # fitted_curve = BondYieldCurve(eval_date, bonds, ytms, curve_function)
            # fitted_curve.plot('CDB Yield Curve by B Spline power of 4')

            today = ytm_date.strftime('%Y%m%d')
            address = f'./results/cdb_ytm_{today}.jpg'
            plt.savefig(address, dpi=600)
            plt.close()
        except Exception as e:
            print(f'Exception:{str(e)} {ytm_date}')
            plot_bond_ylds(ytm_date, bonds, ytms)

        ytm_date += datetime.timedelta(days=1)
    ns_stats.to_csv(os.path.join('./results/', 'ns_stats.csv'), index=False, encoding='utf-8')



if __name__ == '__main__':
    preprocess()
    main()
