import pandas as pd
from fxincome.const import COUPON_TYPE
from fxincome.const import CASHFLOW_TYPE
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta


class Bond:
    def __init__(self, code, initial_date, end_date, issue_price,coupon_rate,coupon_type, coupon_frequency,coupon_period=None):
        self.code=code
        self.initial_date=initial_date
        self.end_date=end_date
        self.face_value=100
        self.issue_price=issue_price#发行价格，附息债券都为100，这个字段主要为贴现债券
        self.coupon_rate=coupon_rate#每100元的利息
        self.coupon_type=coupon_type
        self.coupon_frequency=coupon_frequency
        self.coupon_period=coupon_period if coupon_period else end_date.year-initial_date.year#债券期限（年），如果是整年的债券就可以不输入coupon_period,非整年的就要输入
        self.cashflow_df=self.cal_cashflow()
    def get_dailycoupon(self,date):
        cashflow_df=self.get_cashflow(date,'Undelivered_Lastone')
        if cashflow_df.shape[0]<=1:
            if cashflow_df.iloc[0,0]==date:#1刚好在到期日这天
                dailycoupon=0
                return dailycoupon
            else:#2晚于到期日
                raise Exception('The bond is due'+'('+self.code+')')
        interval_days=(cashflow_df.iloc[1,0]-cashflow_df.iloc[0,0]).days
        if self.coupon_type ==COUPON_TYPE.REGULAR:
            dailycoupon=self.coupon_rate/self.coupon_frequency/interval_days
        elif self.coupon_type==COUPON_TYPE.DUE:
            dailycoupon=self.coupon_rate/interval_days
        elif self.coupon_type==COUPON_TYPE.ZERO:
            dailycoupon=(self.face_value-self.issue_price)/interval_days
        return dailycoupon

    def cal_cashflow(self):
        cashflow_df=pd.DataFrame([],columns=['date','cash'])
        if self.coupon_type==COUPON_TYPE.REGULAR:
            coupon = self.coupon_rate / self.coupon_frequency
            coupon_months = int(12 / self.coupon_frequency)#每隔多少个月付息一次
            coupon_times=int(self.coupon_frequency*(self.coupon_period))#总付息次数
            cashflow_df=cashflow_df.append([{'date':self.initial_date,'cash':-self.face_value}],ignore_index=True)
            for i in range(coupon_times-1):
                date=self.initial_date+relativedelta(months=coupon_months*(i+1))
                cashflow_df=cashflow_df.append([{'date':date,'cash':coupon}],ignore_index=True)
            cashflow_df=cashflow_df.append([{'date':self.end_date,'cash':self.face_value+coupon}],ignore_index=True)
        elif self.coupon_type==COUPON_TYPE.ZERO:
            cashflow_df=cashflow_df.append([{'date':self.initial_date,'cash':-self.issue_price}],ignore_index=True)
            cashflow_df=cashflow_df.append([{'date':self.end_date,'cash':self.face_value}],ignore_index=True)
        elif self.coupon_type==COUPON_TYPE.DUE:
            cashflow_df=cashflow_df.append([{'date':self.initial_date,'cash':-self.face_value}],ignore_index=True)
            cashflow_df=cashflow_df.append([{'date':self.end_date,'cash':self.coupon_rate+self.face_value}],ignore_index=True)
        else:
            raise NotImplementedError("Unknown COUPON_TYPE")
        return cashflow_df
    def get_cashflow(self,date,cashflow_type='Undelivered'):
        cashflow_df=self.cashflow_df
        if cashflow_type==CASHFLOW_TYPE.Undelivered:
            cashflow_df=cashflow_df[cashflow_df['date']>date].copy()
        elif cashflow_type==CASHFLOW_TYPE.History:
            cashflow_df=cashflow_df[cashflow_df['date']<=date].copy()
        elif cashflow_type==CASHFLOW_TYPE.Undelivered_Lastone:
            begin_num=sum(cashflow_df['date'].apply(lambda x:x<=date))-1
            cashflow_df=cashflow_df.iloc[begin_num:,:].copy()
        else:
            raise NotImplementedError("Unknown CASHFLOW_TYPE")
        return cashflow_df

    def ytm_to_dirtyprice(self, date, ytm, full_info=False):
        cashflow_df=self.get_cashflow(date,'Undelivered_Lastone')
        if cashflow_df.shape[0]<=1:
            if cashflow_df.iloc[0,0]==date:#1刚好在到期日这天
                dirtyprice=100
                return dirtyprice
            else:#2晚于到期日
                raise Exception('The bond is due'+'('+self.code+')')
        #3到期日之前
        interval_days=(cashflow_df.iloc[1,0]-cashflow_df.iloc[0,0]).days#当前付息区间的天数
        t=(cashflow_df.iloc[1,0]-date).days#距下一个付息日的天数
        years_days=365#todo 怎么定366还是365需要在研究，现在暂时不纠结这个细节
        cashflow_df=cashflow_df.iloc[1:,:]
        if self.coupon_type==COUPON_TYPE.REGULAR:
            if cashflow_df.shape[0]>1:
                cashflow_df['deflator']=[ 1/(1+ytm/100/self.coupon_frequency)**(t/interval_days+i) for i in range(cashflow_df.shape[0])]
            elif cashflow_df.shape[0]==1:
                cashflow_df['deflator']=[1/(1+ytm/100*t/years_days)]
        elif self.coupon_type in [COUPON_TYPE.ZERO,COUPON_TYPE.DUE]:
            cashflow_df['deflator']=[1/(1+ytm/100*t/years_days)]
        cashflow_df['cash_deflated']=cashflow_df['cash']*cashflow_df['deflator']
        dirtyprice=cashflow_df['cash_deflated'].sum()
        #todo cashflow_df可能是个有用的datafram，后面要用再来拿
        if full_info:
            return cashflow_df
        else:
            return dirtyprice

    def dirtyprice_to_ytm(self, date, dirtyprice):
        ytm=5.0
        dirtyprice_cal=self.ytm_to_dirtyprice(date, ytm)
        while abs(dirtyprice_cal-dirtyprice)>0.00001:
            k= (self.ytm_to_dirtyprice(date, ytm + 0.00005) - self.ytm_to_dirtyprice(date, ytm - 0.00005)) / 0.0001
            b=dirtyprice_cal-k*ytm
            ytm=(dirtyprice-b)/k
            dirtyprice_cal=self.ytm_to_dirtyprice(date, ytm)
        return ytm

    def cal_accrued_interest(self,date):
        cashflow_df=self.get_cashflow(date,'Undelivered_Lastone')
        if cashflow_df.shape[0]<=1:
            if cashflow_df.shape[0]<=1:
                if cashflow_df.iloc[0,0]==date:#1刚好在到期日这天
                    accrued_interest=0
                    return accrued_interest
                else:#2晚于到期日
                    raise Exception('The bond is due'+'('+self.code+')')
        interval_days=(cashflow_df.iloc[1,0]-cashflow_df.iloc[0,0]).days#当前付息区间的天数
        t=(date-cashflow_df.iloc[0,0]).days#距上一个付息日的天数
        if self.coupon_type==COUPON_TYPE.REGULAR:
            accrued_interest=self.coupon_rate/self.coupon_frequency*t/interval_days
        elif self.coupon_type==COUPON_TYPE.DUE:
            accrued_interest=self.coupon_rate*t/interval_days
        elif self.coupon_type==COUPON_TYPE.ZERO:
            accrued_interest=(self.face_value-self.issue_price)*t/interval_days
        return accrued_interest

    def dirtyprice_to_cleanprice(self, date, dirtyprice):
        accrued_interest=self.cal_accrued_interest(date)
        return dirtyprice-accrued_interest
    def cleanprice_to_dirtyprice(self, date, cleanprice):
        accrued_interest=self.cal_accrued_interest(date)
        return cleanprice+accrued_interest
    def ytm_to_cleanprice(self, date, ytm):
        dirtyprice=self.ytm_to_dirtyprice(date, ytm)
        cleanprice=self.dirtyprice_to_cleanprice(date, dirtyprice)
        return cleanprice
    def cleanprice_to_ytm(self, date, cleanprice):
        dirtyprice=self.cleanprice_to_dirtyprice(date, cleanprice)
        ytm=self.dirtyprice_to_ytm(date, dirtyprice)
        return ytm
    def curve_to_ytm(self, date, curve_df):
        days=(self.cashflow_df.iloc[-1,0]-date).days
        begin_num=sum(curve_df['days'].apply(lambda x:x<=days))-1
        curve_df=curve_df.iloc[begin_num:begin_num+2,:]
        if days<0:
            return None
        elif curve_df.iloc[-1,0]==days:
            return curve_df.iloc[-1,1]
        elif curve_df.shape[0]<2:
            raise Exception('The curve is too short'+'('+self.code+')')
        else:
            ytm=(curve_df.iloc[1,1]-curve_df.iloc[0,1])/(curve_df.iloc[1,0]-curve_df.iloc[0,0])*(days-curve_df.iloc[0,0])+curve_df.iloc[0,1]
            return ytm

    def curve_to_dirtyprice(self, date, curve_df):
        ytm=self.curve_to_ytm(date, curve_df)
        dirtyprice=self.ytm_to_dirtyprice(date, ytm)
        return dirtyprice

    def curve_to_cleanprice(self, date, curve_df):
        ytm=self.curve_to_ytm(date, curve_df)
        cleanprice=self.ytm_to_cleanprice(date, ytm)
        return cleanprice
    def RealDailyR_to_AmortizedPrice(self, date, RealDailyR):
        RealDailyR=RealDailyR/100
        cashflow_date_df=list(self.get_cashflow(date,'Undelivered_Lastone')['date'])
        if self.coupon_type ==COUPON_TYPE.REGULAR:
            coupon=self.coupon_rate/self.coupon_frequency
        elif self.coupon_type==COUPON_TYPE.DUE:
            coupon=self.coupon_rate
        elif self.coupon_type==COUPON_TYPE.ZERO:
            coupon=(self.face_value-self.issue_price)
        else:
            raise NotImplementedError("Unknown COUPON_TYPE")
        Tlist = []
        tlist = []
        nlist = []
        n = 0
        for date_i in cashflow_date_df[1:]:
            Tlist.append((cashflow_date_df[-1] - date_i).days)
            tlist.append((date_i - cashflow_date_df[n]).days)
            nlist.append((date_i - cashflow_date_df[n]).days)
            n += 1
        nlist[0]=(cashflow_date_df[1]-date).days
        Tlist = np.array(Tlist)
        tlist = np.array(tlist)
        nlist = np.array(nlist)
        price_coupon = sum(
            (coupon / tlist * ((1 + RealDailyR) ** nlist - 1) / RealDailyR) * (1 + RealDailyR) ** Tlist)
        AmortizedPrice=(100+price_coupon)/((1+RealDailyR)**sum(nlist))
        return AmortizedPrice

    def AmortizedPrice_to_RealDailyR(self, date, AmortizedPrice):
        RealDailyR = 5 / 365
        price=self.RealDailyR_to_AmortizedPrice(date, RealDailyR)
        while abs(price-AmortizedPrice)>0.000001:
            k= (self.RealDailyR_to_AmortizedPrice(date, RealDailyR + 0.00000001) -
                self.RealDailyR_to_AmortizedPrice(date, RealDailyR - 0.00000001)) / 0.00000002
            b=price-k*RealDailyR
            RealDailyR=(AmortizedPrice-b)/k
            price=self.RealDailyR_to_AmortizedPrice(date, RealDailyR)
        return RealDailyR
    def ytm_to_dv01(self, date, ytm):
        if not ytm:
            return np.nan
        elif date<self.end_date:
            dirtyprice_up=self.ytm_to_dirtyprice(date, ytm + 0.005)
            dirtyprice_down=self.ytm_to_dirtyprice(date, ytm - 0.005)
            dv01=(dirtyprice_up-dirtyprice_down)
            return dv01
        else:
            return 0
    def curve_to_dv01(self, date, curve_df):
        ytm=self.curve_to_ytm(date, curve_df)
        dv01=self.ytm_to_dv01(date, ytm)
        return dv01

    def ytm_to_duration(self, date, ytm, DURARION_TYPE):
        if not ytm:
            return np.nan
        elif date<self.end_date:
            cashflow_df=self.ytm_to_dirtyprice(date, ytm, True)
            cashflow_df['cash_deflated_days']=cashflow_df.apply(lambda x:(x['date']-date).days*x['cash_deflated']/365,axis=1)
            duration=cashflow_df['cash_deflated_days'].sum()/cashflow_df['cash_deflated'].sum()
            if DURARION_TYPE=='Macaulay':
                pass
            elif DURARION_TYPE=='Modified':
                duration=duration/(1+ytm/100)
            else:
                raise NotImplementedError("Unknown DURARION_TYPE")
            return duration
        else:
            return 0
    def curve_to_duration(self, date, curve_df, DURARION_TYPE):
        ytm=self.curve_to_ytm(date, curve_df)
        duration=self.ytm_to_duration(date, ytm, DURARION_TYPE)
        return duration






if __name__=='__main__':
    code='200016'
    initial_date=datetime.datetime(2020,11,19)
    end_date=datetime.datetime(2030,11,19)
    issue_price=100
    coupon_rate=3.27
    coupon_type='附息'
    coupon_frequency=2
    a=Bond(code, initial_date, end_date,issue_price, coupon_rate,coupon_type, coupon_frequency)
    date=datetime.datetime(2021,5,12)
    ytm=3.1450
    dirtyprice=102.5927
    cleanprice=101.0210
    print(a.cal_accrued_interest(date))

    print(a.ytm_to_dirtyprice(date, ytm))
    print(a.ytm_to_cleanprice(date, ytm))

    print(a.dirtyprice_to_ytm(date, dirtyprice))
    print(a.dirtyprice_to_cleanprice(date, dirtyprice))###

    print(a.cleanprice_to_ytm(date, cleanprice))
    print(a.cleanprice_to_dirtyprice(date, cleanprice))

    # x=pd.DataFrame([[1],[2]],columns=['123'])
    # x['111']=[1,2]
    # print(x)
    x=pd.DataFrame([],columns=['1','2'])
    x=x.append([1,2])
    print(x)

